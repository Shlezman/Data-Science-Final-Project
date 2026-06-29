"""SentiSense live UI backend — FastAPI REST + websocket, serves the built React SPA on :3000.

Reuses everything: DB queries (``ui.queries``), the champion config + metrics
(``sentisense.serve.champion``), and the mirofish agent-graph (``sentisense.sim.graph_api``).
Reads the daily orchestrator's status JSON for the health view. No metric is reinvented —
the confusion matrix comes from ``model_predictions`` (predicted vs actual).

Run (server-side, inside /tf, port 3000 exposed to host):
    uv run --extra ui --extra finance --extra ml python -m ui.app
    # or: uv run --extra ui ... uvicorn ui.app:app --host 0.0.0.0 --port 3000
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from sqlalchemy import text

from sentisense.constants import REPO_ROOT
from sentisense.db import get_engine
from ui import queries

_STATUS_PATH = REPO_ROOT / "logs" / "daily_live_status.json"
_DIST = REPO_ROOT / "ui" / "frontend" / "dist"
_MIRO_BASE = os.environ.get("SENTISENSE_MIRO_BASE_URL", "http://localhost:5001")

app = FastAPI(title="SentiSense live", version="1.0")


def _sim_modes() -> list[str]:
    """Available simulation modes (mirofish config; safe default if import fails)."""
    try:
        from sentisense.sim.config import SIM_MODES
        return list(SIM_MODES)
    except Exception:  # noqa: BLE001
        return ["source", "flat"]


@app.get("/api/health")
def health() -> dict:
    """Last orchestrator run status + the served champion version."""
    from sentisense.serve.champion import load_champion

    status = {}
    if _STATUS_PATH.exists():
        try:
            status = json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            status = {"error": "unreadable status file"}
    return {"ok": True, "champion": load_champion().get("version"), "last_run": status}


@app.get("/api/dashboard")
def dashboard() -> dict:
    """Champion accuracy + confusion matrix (settled predictions) + live last-day headlines."""
    from sentisense.serve.champion import load_champion

    champ = load_champion()
    rows = queries.prediction_rows(version=champ.get("version"))
    cm = queries.confusion_matrix(rows)
    day = queries.latest_date()
    latest = queries.headlines_for_date(day=day, page=0, page_size=100) if day else {"headlines": []}
    recent = [{"date": str(r["date"]), "prediction": bool(r["prediction"]),
               "confidence": round(float(r["confidence"]), 4),
               "actual": (None if r["actual"] is None else bool(r["actual"]))}
              for r in rows[:60]]
    return {"champion": champ.get("version"), "confusion": cm, "recent": recent,
            "latest_headlines": latest}


@app.get("/api/headlines/latest")
def headlines_latest(page: int = Query(0, ge=0), page_size: int = Query(50, ge=1, le=200)) -> dict:
    """Headlines for the most recent stored date (dashboard live ticker)."""
    day = queries.latest_date()
    if day is None:
        return {"headlines": [], "total": 0}
    return queries.headlines_for_date(day=day, page=page, page_size=page_size)


@app.get("/api/headlines")
def headlines(date: str, page: int = Query(0, ge=0), page_size: int = Query(50, ge=1, le=200)) -> dict:
    """Paginated headlines for a given date (archive)."""
    return queries.headlines_for_date(day=date, page=page, page_size=page_size)


@app.get("/api/dates")
def dates(page: int = Query(0, ge=0), page_size: int = Query(60, ge=1, le=400)) -> dict:
    """Distinct headline dates, newest first (archive date list)."""
    return {"dates": queries.available_dates(page=page, page_size=page_size)}


@app.get("/api/sim/modes")
def sim_modes() -> dict:
    """Selectable simulation modes."""
    return {"modes": _sim_modes()}


@app.get("/api/sim/dates")
def sim_dates() -> dict:
    """Dates that have a cached narrative simulation (newest first)."""
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT sim_date FROM narrative_sim ORDER BY sim_date DESC LIMIT 400")).all()
    return {"dates": [str(r[0]) for r in rows]}


@app.get("/api/sim/graph")
def sim_graph(date: str | None = None, mode: str = "source") -> JSONResponse:
    """Cached agent-interaction graph for a date (or the latest) — nodes/edges/meta."""
    from sentisense.sim import graph_api

    g = graph_api.graph_for_date(date, mode=mode) if date else graph_api.latest_graph(mode=mode)
    if not g:
        return JSONResponse({"error": "no simulation graph for that date/mode"}, status_code=404)
    return JSONResponse(g)


@app.get("/api/sim/report")
def sim_report(date: str, mode: str = "source") -> JSONResponse:
    """Cached narrative report (markdown + sections) for a date."""
    from sentisense.sim import graph_api

    r = graph_api.report_for_date(date, mode=mode)
    if not r:
        return JSONResponse({"error": "no report for that date/mode"}, status_code=404)
    return JSONResponse(r)


@app.websocket("/ws/sim/run")
async def ws_sim_run(ws: WebSocket) -> None:
    """Trigger a sim run and stream coarse progress + the final graph.

    Mirofish is poll-based (no true step stream), so we emit a running heartbeat while the
    blocking ``run_day`` executes in a worker thread, then push the resulting graph. If a date
    is already cached, the graph returns immediately. Errors (e.g. MiroFish service down) are
    sent as an ``error`` event rather than dropping the socket.
    """
    await ws.accept()
    try:
        req = await ws.receive_json()
        date, mode = req.get("date"), req.get("mode", "source")
        await ws.send_json({"event": "accepted", "date": date, "mode": mode})

        from sentisense.sim import graph_api
        cached = graph_api.graph_for_date(date, mode=mode) if date else None
        if cached:
            await ws.send_json({"event": "done", "cached": True, "graph": cached})
            return

        async def _heartbeat():
            i = 0
            while True:
                await asyncio.sleep(5)
                i += 1
                await ws.send_json({"event": "running", "elapsed_s": i * 5})

        hb = asyncio.create_task(_heartbeat())
        try:
            from sentisense.sim.miro_client import MiroClient
            from sentisense.sim.runner import run_day

            def _run():
                client = MiroClient(base_url=_MIRO_BASE)
                run_day(client, get_engine(), date, mode=mode)
                g = graph_api.graph_for_date(date, mode=mode)
                return g

            graph = await asyncio.to_thread(_run)
        finally:
            hb.cancel()

        if graph:
            await ws.send_json({"event": "done", "cached": False, "graph": graph})
        else:
            await ws.send_json({"event": "error", "message": "sim produced no graph"})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001 — surface failures to the client, keep server up
        logger.warning("sim ws run failed: {}", str(exc)[:200])
        try:
            await ws.send_json({"event": "error", "message": str(exc)[:300]})
        except Exception:  # noqa: BLE001
            pass


if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
else:
    @app.get("/")
    def _no_build() -> JSONResponse:
        return JSONResponse({"error": "SPA not built. Run: cd ui/frontend && npm install && npm run build",
                             "api": "/api/health"}, status_code=200)


def main() -> None:
    """Serve on 0.0.0.0:3000 (override with SENTISENSE_UI_PORT)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SENTISENSE_UI_PORT", "3000")))


if __name__ == "__main__":
    main()
