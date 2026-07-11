"""Phase 5 acceptance: the FastAPI app replays a fixture race and every
WebSocket message validates against the pydantic schemas; REST endpoints and
/api/simulate work against the live engine state."""

import json

import pytest
from fastapi.testclient import TestClient

from app import schemas
from app.api import create_app
from tests.test_backtest import make_race_dir


@pytest.fixture(scope="module")
def client(monza, tiny_model, tmp_path_factory):
    model, _ = tiny_model
    race_dir = make_race_dir(tmp_path_factory.mktemp("race"), monza)
    app = create_app(race_dir=race_dir, speed=1200.0, model=model)
    with TestClient(app) as c:
        yield c


def test_ws_messages_validate(client):
    """Consume the stream mid-replay; every message must match the schema and
    carry monotonically increasing _id."""
    seen, last_id = 0, None
    with client.websocket_connect("/ws/live") as ws:
        while seen < 25:
            msg = json.loads(ws.receive_text())
            update = schemas.LapUpdate.model_validate(msg)
            assert update.type == "lap_update"
            assert 1 <= update.lap <= update.laps_total
            if last_id is not None:
                assert msg["_id"] > last_id
            last_id = msg["_id"]
            seen += 1
    assert seen == 25


def test_ws_resume_from_last_id(client):
    with client.websocket_connect("/ws/live") as ws:
        first = json.loads(ws.receive_text())
    with client.websocket_connect(f"/ws/live?since={first['_id']}") as ws:
        nxt = json.loads(ws.receive_text())
        assert nxt["_id"] > first["_id"]


def test_rest_endpoints(client):
    session = client.get("/api/session").json()
    assert session["track_id"] == "monza" and session["data_source"] == "replay"
    schemas.SessionInfo.model_validate(session)

    state = client.get("/api/state")
    assert state.status_code == 200
    schemas.LapUpdate.model_validate(state.json())

    preds = client.get("/api/predictions").json()
    assert isinstance(preds, dict) and len(preds) >= 10

    health = client.get("/api/health").json()
    assert health["status"] == "ok"


def test_simulate_endpoint(client):
    # wait until a top driver has some laps behind them
    import time

    driver = None
    for _ in range(60):
        state = client.get("/api/state").json()
        drivers = state.get("drivers") or []
        ready = [d for d in drivers if (d.get("lap_number") or 0) > 8 and d.get("compound")]
        if ready:
            driver = ready[0]["driver_number"]
            break
        time.sleep(0.3)
    assert driver is not None, "no driver progressed far enough to simulate"

    resp = client.post("/api/simulate", json={"driver_number": driver, "n_sims": 300})
    assert resp.status_code == 200, resp.text
    body = schemas.SimulateResponse.model_validate(resp.json())
    assert body.baseline.finish_time_s.p10 <= body.baseline.finish_time_s.p90
    assert len(body.strategies) >= 5
    assert body.strategies == sorted(body.strategies, key=lambda s: s.expected_position)


def test_simulate_unknown_driver_422(client):
    resp = client.post("/api/simulate", json={"driver_number": 99, "n_sims": 200})
    assert resp.status_code == 422
