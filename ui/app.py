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
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
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
_SIM_HEALTH_TTL = 30.0
_sim_health: dict = {"t": -1e9, "val": None}

# --- site login gate ---------------------------------------------------------
# One shared password from the environment (never committed). When unset, the
# gate is OFF (local dev). The session cookie is an HMAC of a fixed message
# under a key derived from the password, so it is stateless and survives
# restarts; rotating the password invalidates every session.
_UI_PASSWORD = os.environ.get("SENTISENSE_UI_PASSWORD", "")
_AUTH_COOKIE = "ss_auth"


def _auth_token() -> str:
    """The expected session-cookie value for the current password."""
    key = hashlib.sha256(("sentisense-ui:" + _UI_PASSWORD).encode()).digest()
    return hmac.new(key, b"session-v1", hashlib.sha256).hexdigest()


def _is_authed(cookies) -> bool:
    """True when the gate is off or the request carries a valid session cookie."""
    if not _UI_PASSWORD:
        return True
    tok = cookies.get(_AUTH_COOKIE, "")
    return bool(tok) and hmac.compare_digest(tok, _auth_token())


app = FastAPI(title="SentiSense live", version="1.0")

_AUTH_EXEMPT = ("/api/login", "/api/auth")


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Require the session cookie for every data endpoint (API + websocket upgrade).

    Static assets and the SPA shell stay open — the shell renders the login
    screen; only data is gated.
    """
    path = request.url.path
    if (path.startswith("/api") or path.startswith("/ws")) and path not in _AUTH_EXEMPT:
        if not _is_authed(request.cookies):
            return JSONResponse({"error": "authentication required"}, status_code=401)
    return await call_next(request)


@app.get("/api/auth")
def auth_state(request: Request) -> dict:
    """Whether this browser session is authenticated (drives the login screen)."""
    return {"authed": _is_authed(request.cookies), "gated": bool(_UI_PASSWORD)}


@app.post("/api/login")
async def login(request: Request) -> JSONResponse:
    """Validate the shared password and set the session cookie (30 days)."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    supplied = str(body.get("password", ""))
    if not _UI_PASSWORD or not secrets.compare_digest(supplied, _UI_PASSWORD):
        return JSONResponse({"error": "wrong password"}, status_code=401)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(_AUTH_COOKIE, _auth_token(), max_age=30 * 24 * 3600,
                    httponly=True, samesite="lax")
    return resp

_CACHE: dict = {}
_CACHE_TTL = 60.0


def _cached(key: str, fn):
    """Memoise a read-only endpoint result for ``_CACHE_TTL`` seconds (per key)."""
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and now - hit[0] < _CACHE_TTL:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


def _sim_modes() -> list[str]:
    """Available simulation modes (mirofish config; safe default if import fails)."""
    try:
        from sentisense.sim.config import SIM_MODES
        return list(SIM_MODES)
    except Exception:  # noqa: BLE001
        return ["source", "flat"]


@app.get("/api/health")
def health() -> dict:
    """Last orchestrator run status + the ACTIVE served model (registry winner, else pinned)."""
    status = {}
    if _STATUS_PATH.exists():
        try:
            status = json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            status = {"error": "unreadable status file"}
    version, model_type = _active_served()
    return {"ok": True, "champion": version, "model_type": model_type, "last_run": status}


def _active_served() -> tuple[str, str]:
    """(version, model_type) currently serving — active registry model, else pinned champion."""
    from sentisense.serve.champion import load_champion

    try:
        from sentisense.serve import registry
        active = registry.get_active()
        if active:
            return active["version"], active["model_type"]
    except Exception:  # noqa: BLE001 — registry table may not exist yet
        pass
    return load_champion().get("version"), "pinned"


@app.get("/api/dashboard")
def dashboard() -> dict:
    """Served-model accuracy + live metrics (its predictions) + live last-day headlines.

    When the ACTIVE model is freshly promoted it has no prediction rows yet — fall back to the
    all-model prediction history (``history_scope='all'``) so the recent table and live metrics
    aren't empty until the new champion writes its first daily row.
    """
    version, model_type = _active_served()
    active_rows = queries.prediction_rows(version=version)
    rows, history_scope = active_rows, "active"
    if not rows:
        rows = queries.prediction_rows(version=None)
        history_scope = "all"
    cm = queries.confusion_matrix(rows)
    ev = queries.active_model_metrics()

    # Cumulative score: seed with the model's held-out evaluation, then fold in each settled
    # LIVE day of the SAME model (never other versions' history — that would launder lineage).
    combined = None
    if ev and ev.get("accuracy") is not None and ev.get("n"):
        cm_active = queries.confusion_matrix(active_rows) if active_rows else None
        live_ok = (cm_active["tp"] + cm_active["tn"]) if cm_active else 0
        live_n = cm_active["n"] if cm_active else 0
        n_all = ev["n"] + live_n
        combined = {"accuracy": round((ev["accuracy"] * ev["n"] + live_ok) / n_all, 4),
                    "n": n_all, "n_eval": ev["n"], "n_live": live_n}

    day = queries.latest_date()
    latest = queries.headlines_for_date(day=day, page=0, page_size=100) if day else {"headlines": []}
    recent = [{"date": str(r["date"]), "prediction": bool(r["prediction"]),
               "confidence": round(float(r["confidence"]), 4),
               "actual": (None if r["actual"] is None else bool(r["actual"]))}
              for r in rows[:60]]
    return {"champion": version, "model_type": model_type, "confusion": cm, "recent": recent,
            "history_scope": history_scope, "combined": combined,
            "eval_metrics": ev, "latest_headlines": latest}


@app.get("/api/prediction/today")
def prediction_today() -> dict:
    """Current-day served prediction (up/down + confidence) for the dashboard hero."""
    return _cached("today", lambda: queries.today_prediction() or {})


_PERF_OVERRIDE = REPO_ROOT / "models" / "performance.json"


def _build_performance() -> dict:
    """Compute the FULL Model-performance panel payload server-side.

    The UI renders this JSON verbatim, so the panel can be changed manually (drop an edited
    copy at ``models/performance.json`` — it wins over the computed values) or methodically
    (regenerate it with ``scripts/generate_performance.py``).
    """
    version, model_type = _active_served()
    active_rows = queries.prediction_rows(version=version)
    rows = active_rows or queries.prediction_rows(version=None)
    cm = queries.confusion_matrix(rows)
    ev = queries.active_model_metrics() or {}

    cm_active = queries.confusion_matrix(active_rows) if active_rows else None
    live_n = cm_active["n"] if cm_active else 0
    live_ok = (cm_active["tp"] + cm_active["tn"]) if cm_active else 0
    ev_n = ev.get("n") or 0
    n_all = ev_n + live_n
    acc = (((ev.get("accuracy") or 0) * ev_n + live_ok) / n_all) if n_all else cm.get("accuracy")
    mcc = cm_active["mcc"] if (cm_active and live_n > 0) else ev.get("mcc")

    def pctf(v):
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "—"

    return {
        "source": "computed",
        "champion": version,
        "model_type": model_type,
        "subtitle": "Evaluation and live-monitoring metrics with reference baselines.",
        "core_tag": "Overall · evaluation + live",
        "core": [
            {"label": "Accuracy", "value": (round(acc, 4) if acc is not None else None),
             "kind": "accuracy", "baseline": 0.5, "domain": [0, 1], "scope": "Overall",
             "comparison": (f"Eval {pctf(ev.get('accuracy'))}" if ev.get("accuracy") is not None else None),
             "info": "The share of predictions that matched the actual market direction."},
            {"label": "ROC-AUC", "value": ev.get("roc_auc"),
             "kind": "auc", "baseline": 0.5, "domain": [0, 1], "scope": "Evaluation",
             "comparison": "Higher is better",
             "info": "How well the model separates up days from down days across decision thresholds."},
            {"label": "MCC", "value": (round(mcc, 4) if isinstance(mcc, (int, float)) else None),
             "kind": "mcc", "baseline": 0, "domain": [-1, 1],
             "scope": ("Live" if live_n > 0 else "Evaluation"),
             "comparison": (f"Eval {ev.get('mcc')}" if live_n > 0 and ev.get("mcc") is not None
                            else "Range −1 to +1"),
             "info": "A balanced correlation score from −1 to +1; zero means no predictive relationship."},
        ],
        "classification_tag": (f"Live monitoring · {live_n} days" if live_n > 0 else None),
        "classification": [
            {"label": "Precision", "value": (cm_active["precision"] if live_n > 0 else None),
             "accent": "#2dd4bf",
             "info": "Of the predicted positive days, the share that were actually positive."},
            {"label": "Recall", "value": (cm_active["recall"] if live_n > 0 else None),
             "accent": "#a78bfa",
             "info": "Of the actual positive days, the share the model identified."},
            {"label": "F1", "value": (cm_active["f1"] if live_n > 0 else None),
             "accent": "#fbbf24",
             "info": "The harmonic mean of precision and recall."},
        ],
        "sample": {"total": n_all or cm.get("n", 0), "eval": ev_n, "live": live_n,
                   "pending": cm.get("pending", 0)},
    }


# Versioned performance documents live in the front machine's MongoDB (env-configured URL,
# e.g. mongodb://user:pass@localhost:21771/?authSource=admin). The ACTIVE Mongo version wins
# over the file override, which wins over computed values — so the panel can be tuned,
# versioned, and rolled back without touching git.
_MONGO_URL = os.environ.get("SENTISENSE_MONGO_URL", "")


def _perf_coll():
    """The performance_versions Mongo collection, or None (unset env / driver / server down)."""
    if not _MONGO_URL:
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(_MONGO_URL, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        return client["sentisense"]["performance_versions"]
    except Exception as exc:  # noqa: BLE001 — Mongo is optional; never break the panel
        logger.warning("Mongo unavailable ({}); performance versions disabled.", str(exc)[:120])
        return None


@app.get("/api/performance")
def performance() -> dict:
    """The Model-performance panel as one JSON document.

    Resolution order: active Mongo version > models/performance.json > computed.
    """
    coll = _perf_coll()
    if coll is not None:
        try:
            row = coll.find_one({"active": True})
            if row and isinstance(row.get("doc"), dict):
                doc = dict(row["doc"])
                doc["source"] = f"mongo:{row['_id']}"
                return doc
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mongo active-version read failed: {}", str(exc)[:120])
    if _PERF_OVERRIDE.exists():
        try:
            doc = json.loads(_PERF_OVERRIDE.read_text(encoding="utf-8"))
            doc["source"] = "file"
            return doc
        except Exception as exc:  # noqa: BLE001 — a broken override must not blank the panel
            logger.warning("performance.json unreadable ({}); serving computed.", str(exc)[:120])
    try:
        return _cached("performance", _build_performance)
    except Exception as exc:  # noqa: BLE001
        logger.warning("/api/performance failed: {}", str(exc)[:300])
        return {"source": "error", "error": str(exc)[:200], "core": [], "classification": [],
                "sample": {"total": 0, "eval": 0, "live": 0, "pending": 0}}


@app.get("/api/performance/versions")
def performance_versions() -> dict:
    """List stored performance versions (metadata only; fetch one by id for the doc)."""
    coll = _perf_coll()
    if coll is None:
        return {"versions": [], "mongo": False}
    rows = [{"id": str(r["_id"]), "note": r.get("note", ""), "active": bool(r.get("active")),
             "created_at": str(r.get("created_at", ""))}
            for r in coll.find({}, {"doc": False}).sort("created_at", -1).limit(100)]
    return {"versions": rows, "mongo": True}


@app.get("/api/performance/versions/{vid}")
def performance_version(vid: str) -> JSONResponse:
    """One stored version's full document (for the editor)."""
    coll = _perf_coll()
    if coll is None:
        return JSONResponse({"error": "mongo unavailable"}, status_code=503)
    from bson import ObjectId
    try:
        row = coll.find_one({"_id": ObjectId(vid)})
    except Exception:  # noqa: BLE001 — malformed id
        row = None
    if not row:
        return JSONResponse({"error": "unknown version"}, status_code=404)
    return JSONResponse({"id": vid, "note": row.get("note", ""), "active": bool(row.get("active")),
                         "doc": row.get("doc", {})})


@app.post("/api/performance/versions")
async def performance_version_save(request: Request) -> JSONResponse:
    """Save a new version. Body: {doc?: object, note?: str} — doc defaults to computed."""
    coll = _perf_coll()
    if coll is None:
        return JSONResponse({"error": "mongo unavailable"}, status_code=503)
    import datetime as dt
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    doc = body.get("doc")
    if doc is not None and not isinstance(doc, dict):
        return JSONResponse({"error": "doc must be an object"}, status_code=400)
    if doc is None:
        doc = _build_performance()
    res = coll.insert_one({"doc": doc, "note": str(body.get("note", ""))[:200],
                           "active": False, "created_at": dt.datetime.utcnow()})
    return JSONResponse({"id": str(res.inserted_id), "active": False})


@app.post("/api/performance/versions/{vid}/activate")
def performance_version_activate(vid: str) -> JSONResponse:
    """Make one version the served document (deactivates the rest)."""
    coll = _perf_coll()
    if coll is None:
        return JSONResponse({"error": "mongo unavailable"}, status_code=503)
    from bson import ObjectId
    try:
        oid = ObjectId(vid)
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "bad id"}, status_code=400)
    if not coll.find_one({"_id": oid}):
        return JSONResponse({"error": "unknown version"}, status_code=404)
    coll.update_many({}, {"$set": {"active": False}})
    coll.update_one({"_id": oid}, {"$set": {"active": True}})
    return JSONResponse({"ok": True, "active": vid})


@app.post("/api/performance/versions/deactivate")
def performance_version_deactivate() -> JSONResponse:
    """Deactivate all versions — the panel falls back to file/computed values."""
    coll = _perf_coll()
    if coll is None:
        return JSONResponse({"error": "mongo unavailable"}, status_code=503)
    coll.update_many({}, {"$set": {"active": False}})
    return JSONResponse({"ok": True})


@app.post("/api/llm/ask")
async def llm_ask(request: Request) -> JSONResponse:
    """Queue a question (or day-narration) for the LLM worker on the GPU box.

    The firewall only passes Postgres between the hosts, so the DB is the transport:
    this inserts a row; ``scripts/llm_worker.py`` answers it; the UI polls /api/llm/answer.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    kind = body.get("kind", "ask")
    if kind not in ("ask", "narrate", "simulate"):
        return JSONResponse({"error": "kind must be ask|narrate|simulate"}, status_code=400)
    day = body.get("date") or None
    question = (str(body.get("question", "")).strip() or None)
    if kind == "ask" and not question:
        return JSONResponse({"error": "question required"}, status_code=400)
    if kind == "simulate" and not day:
        return JSONResponse({"error": "date required for simulate"}, status_code=400)
    if question and len(question) > 2000:
        return JSONResponse({"error": "question too long"}, status_code=400)
    rid = queries.llm_submit(kind=kind, day=day, question=question)
    return JSONResponse({"id": rid, "status": "pending"})


@app.get("/api/llm/answer")
def llm_answer(id: int) -> JSONResponse:
    """Poll one queued LLM request; returns status pending|done|error with the answer."""
    row = queries.llm_fetch(request_id=id)
    if row is None:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    return JSONResponse(row)


@app.get("/api/confusion/full")
def confusion_full() -> dict:
    """In-sample confusion matrix over ALL labeled days (scope='all'), from champion_full_eval."""
    def build() -> dict:
        rows = queries.full_eval_rows()
        cm = queries.confusion_matrix(rows)
        version = rows[0]["model_version"] if rows else None
        return {"scope": "all", "model_version": version, **cm}
    try:
        return _cached("confusion_full", build)
    except Exception as exc:  # noqa: BLE001 — table absent until compute_full_eval runs
        return {"scope": "all", "model_version": None, "n": 0, "error": str(exc)[:200]}


@app.get("/api/eda")
def eda() -> dict:
    """EDA aggregates: volume, sentiment time-series/histogram, relevance, category corr, validation."""
    try:
        return _cached("eda", queries.eda_aggregates)
    except Exception as exc:  # noqa: BLE001 — degrade to empty rather than 500
        logger.warning("/api/eda failed: {}", str(exc)[:300])
        return {"error": str(exc)[:200], "volume": [], "sentiment_ts": [], "sentiment_hist": [],
                "relevance_hist": [], "category_corr": {"labels": [], "matrix": []},
                "validation": {"passed": 0, "failed": 0, "rate": 0.0}}


@app.get("/api/centroids")
def centroids() -> dict:
    """Per-day 3D news centroids (embpca_000..002) coloured by actual up/down."""
    try:
        return _cached("centroids", queries.centroid_points)
    except Exception as exc:  # noqa: BLE001 — daily_embedding_derived/champion_full_eval may be absent
        logger.warning("/api/centroids failed: {}", str(exc)[:300])
        return {"points": [], "error": str(exc)[:200]}


@app.get("/api/centroids/day")
def centroids_day(date: str) -> dict:
    """One day's headline cloud projected into the 16-d embpca space + that day's centroid."""
    try:
        return _cached(f"cday:{date}", lambda: queries.day_centroid_points(day=date))
    except Exception as exc:  # noqa: BLE001 — basis/embeddings may be absent on this DB
        logger.warning("/api/centroids/day failed: {}", str(exc)[:300])
        return {"date": date, "points": [], "centroid": None, "error": str(exc)[:200]}


@app.get("/api/personas")
def personas(date: str) -> dict:
    """Per-source persona votes (up/down/neutral by mean sentiment) + model prediction + actual."""
    try:
        return _cached(f"personas:{date}", lambda: queries.persona_votes(day=date))
    except Exception as exc:  # noqa: BLE001
        logger.warning("/api/personas failed: {}", str(exc)[:300])
        return {"date": date, "personas": [], "general": None, "model": None,
                "actual": None, "error": str(exc)[:200]}


@app.get("/api/models")
def models() -> dict:
    """All registered models (metrics + which is active) for the Models tab."""
    try:
        from sentisense.serve import registry
        rows = registry.list_models()
        for r in rows:
            r.pop("feature_cols", None)          # drop the ~970-col list from the payload
        return {"models": rows}
    except Exception as exc:  # noqa: BLE001 — registry table absent → empty list, not a 500
        return {"models": [], "error": str(exc)[:200]}


@app.post("/api/models/{version}/activate")
def activate_model(version: str) -> JSONResponse:
    """Manually set the active (served) model. Manual picks are sticky vs auto-selection."""
    from sentisense.serve import registry
    if not registry.set_active(version=version, by="manual"):
        return JSONResponse({"error": f"model '{version}' not found"}, status_code=404)
    return JSONResponse({"ok": True, "active": version})


@app.get("/api/headlines/latest")
def headlines_latest(page: int = Query(0, ge=0), page_size: int = Query(50, ge=1, le=200)) -> dict:
    """Headlines for the most recent stored date (dashboard live ticker)."""
    day = queries.latest_date()
    if day is None:
        return {"headlines": [], "total": 0}
    return queries.headlines_for_date(day=day, page=page, page_size=page_size)


@app.get("/api/headlines")
def headlines(date: str, page: int = Query(0, ge=0), page_size: int = Query(50, ge=1, le=200),
              q: str | None = Query(None, max_length=200),
              sort: str = Query("time"), order: str = Query("desc"),
              sentiment_min: int | None = Query(None, ge=-10, le=10),
              sentiment_max: int | None = Query(None, ge=-10, le=10),
              category: str | None = Query(None),
              category_min: int | None = Query(None, ge=0, le=10)) -> dict:
    """Paginated headlines for a given date (archive), searchable, filterable and sortable.

    ``q`` searches headline text and source across the entire date server-side.
    The archive previously filtered only the rows already on screen, which on a
    ~780-headline day meant a search covered about 6% of it while presenting the
    result as if it were the whole day.

    ``sort``/``order`` and the score filters work over the whole date for the same
    reason: the scores were rendered on every row but could not be queried, so
    "the most negative headlines that day" or "security only" had no answer.
    """
    if sort not in queries.SORT_KEYS:
        return JSONResponse({"error": f"sort must be one of {list(queries.SORT_KEYS)}"},
                            status_code=400)
    if order not in ("asc", "desc"):
        return JSONResponse({"error": "order must be 'asc' or 'desc'"}, status_code=400)
    if category is not None and category not in queries.CATEGORY_KEYS:
        return JSONResponse({"error": f"category must be one of {list(queries.CATEGORY_KEYS)}"},
                            status_code=400)
    return queries.headlines_for_date(
        day=date, page=page, page_size=page_size, search=q, sort=sort, order=order,
        sentiment_min=sentiment_min, sentiment_max=sentiment_max,
        category=category, category_min=category_min)


@app.get("/api/dates")
def dates(page: int = Query(0, ge=0),
          page_size: int | None = Query(None, ge=1, le=10000)) -> dict:
    """Distinct headline dates, newest first (archive date list).

    Returns EVERY date unless ``page_size`` is given. It used to cap at 60, and the
    archive only ever asked for the first page, so the date picker reached about 1%
    of the dates on record. The response shape is unchanged.
    """
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
            "SELECT DISTINCT sim_date FROM narrative_sim_graph ORDER BY sim_date DESC LIMIT 400")).all()
    return {"dates": [str(r[0]) for r in rows]}


def _probe_miro(timeout: float = 2.0) -> dict:
    """Fast reachability probe of the MiroFish base URL, to gate LIVE sim runs.

    A raw short-timeout GET — deliberately NOT via ``MiroClient``, whose ``assert_local`` guard
    would refuse a remote base before any network call. Any HTTP response (even 404) = reachable.
    """
    try:
        import requests
        r = requests.get(_MIRO_BASE, timeout=timeout)
        return {"reachable": True, "base": _MIRO_BASE, "reason": f"http {r.status_code}"}
    except Exception as exc:  # noqa: BLE001 — unreachable/timeout/missing dep → live runs off
        return {"reachable": False, "base": _MIRO_BASE, "reason": str(exc)[:140]}


@app.get("/api/sim/health")
def sim_health() -> dict:
    """Whether MiroFish is reachable for LIVE runs (cached ~30s). Cached graphs render regardless.

    The Simulator tab uses this to disable the "Run new simulation" control and show a
    historical-only banner when MiroFish is down — rather than surfacing a raw connection error.
    """
    now = time.monotonic()
    if _sim_health["val"] is not None and now - _sim_health["t"] < _SIM_HEALTH_TTL:
        return _sim_health["val"]
    val = _probe_miro()
    _sim_health.update(t=now, val=val)
    return val


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
    if not _is_authed(ws.cookies):        # HTTP middleware doesn't cover websockets
        await ws.close(code=4401)
        return
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
