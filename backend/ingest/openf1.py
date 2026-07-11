"""OpenF1 live client: MQTT stream (paid accounts) with REST-polling fallback.

Field names verified against real API responses saved in
tests/fixtures/openf1/ (2025 Monza race, session_key 9912).

The testable core is OpenF1Mapper: it turns OpenF1 messages (laps, stints,
position, pit, race_control, weather) into the engine-row dicts RaceEngine
consumes, keeping the same semantics as the historical laps parquet:
  - compound/tyre_life come from `stints` (lap ranges + tyre_age_at_start)
  - pit_in from `pit` (lap_number of the stop), pit_out from is_pit_out_lap
  - track_status is rebuilt from race_control flags into fastf1-style digit
    codes ('1' clear, '2' yellow, '4' SC, '5' red, '6' VSC, '7' VSC ending)
  - clocks are unix epoch seconds (only differences matter downstream)

Live messages carry `_id` (monotonic, ordering/resume) and `_key` (same key =
update to the same object, e.g. a lap gaining sectors): MessageLog dedupes.
"""

import asyncio
import datetime as dt
import json
import logging
import ssl
import time

import httpx

from app import config

log = logging.getLogger("pitwall.openf1")

TOKEN_URL = config.env("OPENF1_TOKEN_URL", "https://api.openf1.org/token")
POLL_INTERVAL_S = 4.0  # REST fallback cadence (rate limit: 3 req/s)
MQTT_TOPICS = ["v1/laps", "v1/stints", "v1/position", "v1/pit", "v1/race_control", "v1/weather"]


def _epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


class MessageLog:
    """_id ordering + _key dedupe for live streams. Returns True when the
    payload is new (or a genuine update to a known key)."""

    def __init__(self):
        self.last_id: int | None = None
        self._by_key: dict[str, dict] = {}

    def accept(self, msg: dict) -> bool:
        mid = msg.get("_id")
        if mid is not None:
            if self.last_id is not None and mid <= self.last_id:
                return False
            self.last_id = mid
        key = msg.get("_key")
        if key is not None:
            prev = self._by_key.get(key)
            if prev == msg:
                return False
            self._by_key[key] = msg
        return True


class OpenF1Mapper:
    """OpenF1 messages -> engine rows. Pure state machine, fixture-tested."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.stints: dict[tuple[int, int], dict] = {}  # (driver, stint_number)
        self.positions: dict[int, int] = {}
        self.pit_laps: set[tuple[int, int]] = set()  # (driver, lap_number of stop)
        self.status_events: list[tuple[float, str]] = []  # (epoch, digit code)
        self._active_yellow = False
        self._active_sc: str | None = None  # '4', '6', '7' or None
        self._red = False
        self.emitted: set[tuple[int, int]] = set()
        self.latest_weather: dict | None = None

    # --- per-endpoint handlers ------------------------------------------------

    def on_stint(self, msg: dict) -> None:
        key = (int(msg["driver_number"]), int(msg["stint_number"]))
        self.stints[key] = msg

    def on_position(self, msg: dict) -> None:
        self.positions[int(msg["driver_number"])] = int(msg["position"])

    def on_pit(self, msg: dict) -> None:
        if msg.get("lap_number") is not None:
            self.pit_laps.add((int(msg["driver_number"]), int(msg["lap_number"])))

    def on_race_control(self, msg: dict) -> None:
        t = _epoch(msg.get("date")) or time.time()
        message = (msg.get("message") or "").upper()
        flag = (msg.get("flag") or "").upper()
        category = msg.get("category") or ""
        code = None
        if category == "SafetyCar" or "SAFETY CAR" in message:
            if "VIRTUAL" in message:
                code = "7" if ("END" in message or "ENDING" in message) else "6"
                self._active_sc = None if code == "7" else "6"
            else:
                code = "4"
                self._active_sc = "4"
                if "IN THIS LAP" in message or "ENDING" in message:
                    self._active_sc = None
        elif flag == "RED":
            code, self._red = "5", True
        elif flag in ("YELLOW", "DOUBLE YELLOW"):
            code, self._active_yellow = "2", True
        elif flag in ("GREEN", "CLEAR"):
            code = "1"
            self._active_yellow = False
            self._red = False
            if self._active_sc == "4":
                self._active_sc = None
        if code:
            self.status_events.append((t, code))

    def on_weather(self, msg: dict) -> dict:
        row = {
            "session_id": self.session_id,
            "time_session_s": _epoch(msg.get("date")),
            "air_temp": msg.get("air_temperature"),
            "track_temp": msg.get("track_temperature"),
            "humidity": msg.get("humidity"),
            "pressure": msg.get("pressure"),
            "rainfall": bool(msg.get("rainfall") or 0),
            "wind_speed": msg.get("wind_speed"),
            "wind_direction": msg.get("wind_direction"),
        }
        self.latest_weather = row
        return row

    # --- lap assembly ------------------------------------------------------------

    def _status_during(self, start: float, end: float) -> str:
        codes = [c for t, c in self.status_events if start <= t <= end]
        # carry active conditions into laps with no events of their own
        if self._active_sc:
            codes.append(self._active_sc)
        elif self._red:
            codes.append("5")
        elif self._active_yellow:
            codes.append("2")
        if not codes:
            codes = ["1"]
        seen: list[str] = []
        for c in codes:
            if not seen or seen[-1] != c:
                seen.append(c)
        return "".join(seen)

    def _stint_for(self, driver: int, lap: int) -> tuple[int | None, str | None, int | None]:
        best = None
        for (dn, stint_no), s in self.stints.items():
            if dn != driver:
                continue
            lap_start = s.get("lap_start") or 0
            lap_end = s.get("lap_end") or 9999
            if lap_start <= lap <= lap_end and (best is None or stint_no > best[0]):
                best = (stint_no, s)
        if best is None:
            return None, None, None
        stint_no, s = best
        age0 = s.get("tyre_age_at_start") or 0
        tyre_life = age0 + (lap - (s.get("lap_start") or lap)) + 1
        return stint_no, s.get("compound"), int(tyre_life)

    def on_lap(self, msg: dict) -> dict | None:
        """Returns an engine row once the lap is complete (has lap_duration)."""
        driver = int(msg["driver_number"])
        lap = int(msg["lap_number"])
        if msg.get("lap_duration") is None or (driver, lap) in self.emitted:
            return None
        self.emitted.add((driver, lap))
        start = _epoch(msg.get("date_start"))
        duration = float(msg["lap_duration"])
        end = (start + duration) if start is not None else None
        stint_no, compound, tyre_life = self._stint_for(driver, lap)
        return {
            "session_id": self.session_id,
            "driver_number": driver,
            "driver_code": None,
            "team_id": None,
            "lap_number": lap,
            "stint": stint_no,
            "lap_time_s": duration,
            "lap_start_session_s": start,
            "time_session_s": end,
            "sector1_s": msg.get("duration_sector_1"),
            "sector2_s": msg.get("duration_sector_2"),
            "sector3_s": msg.get("duration_sector_3"),
            "pit_in": (driver, lap) in self.pit_laps,
            "pit_out": bool(msg.get("is_pit_out_lap")),
            "compound": compound,
            "tyre_life": tyre_life,
            "track_status": self._status_during(start or 0, end or time.time()),
            "position": self.positions.get(driver),
            "deleted": False,
            "is_accurate": True,
            "fastf1_generated": False,
        }

    def enrich_driver_meta(self, drivers_msgs: list[dict]) -> dict[int, dict]:
        """drivers endpoint -> {number: {code, team}} for snapshot decoration."""
        out = {}
        for m in drivers_msgs:
            out[int(m["driver_number"])] = {
                "driver_code": m.get("name_acronym"),
                "team_id": (m.get("team_name") or "").lower().replace(" ", "_") or None,
                "team_colour": m.get("team_colour"),
            }
        return out


async def fetch_token(client: httpx.AsyncClient) -> str | None:
    """OAuth2 password-grant token for the paid real-time tier."""
    username = config.env("OPENF1_USERNAME")
    password = config.env("OPENF1_PASSWORD")
    if not username or not password:
        return None
    resp = await client.post(
        TOKEN_URL, data={"username": username, "password": password, "grant_type": "password"}
    )
    resp.raise_for_status()
    return resp.json().get("access_token")


class OpenF1LiveClient:
    """Attaches to the current/next race session and feeds the RaceEngine.

    Prefers the MQTT stream (mqtt.openf1.org:8883, topics v1/<endpoint>);
    falls back to REST polling every POLL_INTERVAL_S. NOTE: the MQTT auth
    handshake needs a paid account and is verified on race day (see runbook);
    every downstream transformation is fixture-tested."""

    def __init__(self, runtime, model=None):
        self.runtime = runtime
        self.model = model
        self.mapper: OpenF1Mapper | None = None
        self.logs = {topic: MessageLog() for topic in MQTT_TOPICS}
        self._last_poll: dict[str, str] = {}

    async def run(self) -> None:
        from app.state import RaceEngine, SessionMeta
        from app.tracks import load_tracks
        from ingest.fastf1_pull import canon_track, slugify

        async with httpx.AsyncClient(base_url=config.OPENF1_BASE_URL, timeout=20) as client:
            session = await self._wait_for_session(client)
            sk = session["session_key"]
            track_id = canon_track(slugify(session.get("location") or ""))
            tracks = load_tracks()
            track = tracks.get(track_id, {})
            meta = SessionMeta(
                session_id=f"live_{sk}",
                track_id=track_id,
                laps_total=int(track.get("laps_total", 60)),
                lap_length_km=float(track.get("lap_length_km", 5.0)),
                abrasiveness=float(track.get("abrasiveness", 3.0)),
                season=int(session.get("year") or dt.date.today().year),
                event_name=session.get("circuit_short_name"),
                pit_loss_s=float(track.get("pit_loss_s", 22.0)),
                sc_hazard_per_lap=float(track.get("sc_hazard_per_lap", 0.0035)),
                overtake_difficulty=float(track.get("overtake_difficulty", 0.5)),
            )
            self.mapper = OpenF1Mapper(meta.session_id)
            engine = RaceEngine(meta, self.model)
            self.runtime.engine = engine
            log.info("live session attached: %s at %s", sk, track_id)

            token = None
            try:
                token = await fetch_token(client)
            except Exception as exc:  # noqa: BLE001 — fall back to REST
                log.warning("token fetch failed (%s); REST polling only", exc)

            if token:
                try:
                    await self._run_mqtt(sk, token)
                    return
                except Exception as exc:  # noqa: BLE001 — fall back to REST
                    log.warning("MQTT failed (%s); falling back to REST polling", exc)
            await self._poll_rest(client, sk)

    async def _wait_for_session(self, client: httpx.AsyncClient) -> dict:
        while True:
            now = dt.datetime.now(dt.UTC)
            resp = await client.get("/sessions", params={"session_name": "Race", "year": now.year})
            for sess in resp.json() if resp.status_code == 200 else []:
                start = _epoch(sess.get("date_start"))
                end = _epoch(sess.get("date_end"))
                if start and start - 3600 <= now.timestamp() <= (end or start + 7200) + 3600:
                    return sess
            log.info("no live race session; retrying in 60s")
            await asyncio.sleep(60)

    def _dispatch(self, topic: str, msg: dict) -> None:
        if not self.logs.setdefault(topic, MessageLog()).accept(msg):
            return
        mapper, runtime = self.mapper, self.runtime
        kind = topic.rsplit("/", 1)[-1]
        if kind == "laps":
            row = mapper.on_lap(msg)
            if row is not None:
                update = runtime.engine.on_lap(row)
                if update is not None:
                    runtime.publish(update)
        elif kind == "stints":
            mapper.on_stint(msg)
        elif kind == "position":
            mapper.on_position(msg)
        elif kind == "pit":
            mapper.on_pit(msg)
        elif kind == "race_control":
            mapper.on_race_control(msg)
        elif kind == "weather":
            runtime.engine.on_weather(mapper.on_weather(msg))

    async def _run_mqtt(self, session_key: int, token: str) -> None:
        import paho.mqtt.client as mqtt

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_message(_client, _userdata, m):
            try:
                payload = json.loads(m.payload)
                items = payload if isinstance(payload, list) else [payload]
                for item in items:
                    loop.call_soon_threadsafe(queue.put_nowait, (m.topic, item))
            except Exception:  # noqa: BLE001 — never kill the network thread
                log.exception("bad MQTT payload on %s", m.topic)

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.username_pw_set(config.env("OPENF1_USERNAME"), token)
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        client.on_message = on_message
        client.reconnect_delay_set(min_delay=1, max_delay=60)
        client.connect(config.OPENF1_MQTT_HOST, config.OPENF1_MQTT_PORT, keepalive=30)
        for topic in MQTT_TOPICS:
            client.subscribe(topic, qos=1)
        client.loop_start()
        log.info("MQTT connected, subscribed to %s", MQTT_TOPICS)
        try:
            while True:
                topic, msg = await queue.get()
                if msg.get("session_key") in (None, session_key):
                    self._dispatch(topic, msg)
        finally:
            client.loop_stop()
            client.disconnect()

    async def _poll_rest(self, client: httpx.AsyncClient, session_key: int) -> None:
        log.info("REST polling every %.1fs", POLL_INTERVAL_S)
        endpoints = ["stints", "position", "pit", "race_control", "weather", "laps"]
        while True:
            for ep in endpoints:
                params: dict = {"session_key": session_key}
                since = self._last_poll.get(ep)
                if since:
                    params["date>" if ep != "laps" else "date_start>"] = since
                try:
                    resp = await client.get(f"/{ep}", params=params)
                    if resp.status_code != 200:
                        continue
                    rows = resp.json()
                except Exception as exc:  # noqa: BLE001 — keep polling
                    log.warning("poll %s failed: %s", ep, exc)
                    continue
                newest = None
                for msg in rows:
                    self._dispatch(f"v1/{ep}", msg)
                    stamp = msg.get("date") or msg.get("date_start")
                    if stamp and (newest is None or stamp > newest):
                        newest = stamp
                if newest:
                    self._last_poll[ep] = newest
                await asyncio.sleep(POLL_INTERVAL_S / len(endpoints))
