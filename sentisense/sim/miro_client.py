"""Thin HTTP client for the MiroFish service (arm's-length; no MiroFish imports).

Orchestrates the confirmed pipeline (Flask blueprints /api/graph, /api/simulation,
/api/report):

    project ─▶ /api/graph/build ─▶ /api/simulation/create ─▶ /prepare ─▶ /start
            ─▶ /api/report/generate ─▶ /api/report/<id>/sections
            ─▶ /api/graph/data/<graph_id>           (agent graph for the UI)
            ─▶ /api/simulation/interview            (structured agent votes → numeric)

Async stages (build / prepare / start / report.generate) are kicked off then polled via
their status endpoints. Request bodies for project-create / start / interview are marked
CONFIRM — verify against the handler the first time on the box (the routes themselves are
confirmed; only a couple of payload field names are version-tentative).

Requires ``requests`` (the ``finance`` extra already ships it).
"""

from __future__ import annotations

import time

from loguru import logger

from sentisense.sim.config import (
    ALLOW_REMOTE_MIRO,
    ENABLE_REDDIT,
    ENABLE_TWITTER,
    MIRO_URL,
    POLL_SECONDS,
    TIMEOUT_SECONDS,
)
from sentisense.sim.preflight import assert_local


class MiroError(RuntimeError):
    """A MiroFish API call failed or a stage timed out."""


class MiroClient:
    """Stateless-ish HTTP wrapper around a running MiroFish backend."""

    def __init__(self, base_url: str = MIRO_URL, *, poll: int = POLL_SECONDS,
                 timeout: int = TIMEOUT_SECONDS) -> None:
        try:
            import requests  # lazy → importing sentisense.sim never requires it
        except ModuleNotFoundError as exc:
            raise MiroError("sentisense.sim needs 'requests' — `uv sync --extra miro` "
                            "(or pip install requests).") from exc
        assert_local(base_url, allow_remote=ALLOW_REMOTE_MIRO)
        self.base = base_url.rstrip("/")
        self.poll = poll
        self.stage_timeout = timeout
        self._s = requests.Session()

    # ── low-level ────────────────────────────────────────────────────────────
    def _json(self, method: str, path: str, **kw) -> dict:
        r = self._s.request(method, f"{self.base}{path}", timeout=120, **kw)
        if r.status_code >= 400:
            raise MiroError(f"{method} {path} → HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        if isinstance(body, dict) and body.get("success") is False:
            raise MiroError(f"{method} {path} → {body.get('error', body)}")
        return body

    def _poll(self, path: str, *, done, payload: dict | None = None, what: str = "task"):
        """Poll ``path`` until ``done(body)`` is True (or timeout). Returns the final body."""
        deadline = time.time() + self.stage_timeout
        method = "POST" if payload is not None else "GET"
        while time.time() < deadline:
            body = self._json(method, path, json=payload) if payload is not None else self._json("GET", path)
            if done(body):
                return body
            time.sleep(self.poll)
        raise MiroError(f"{what} timed out after {self.stage_timeout}s ({path})")

    @staticmethod
    def _data(body: dict) -> dict:
        return body.get("data", body) if isinstance(body, dict) else {}

    # ── pipeline stages (routes confirmed from MiroFish source) ────────────────
    def create_project(self, seed_text: str, name: str) -> str:
        # CONFIRM payload field names against /api/graph/project (create) handler.
        body = self._json("POST", "/api/graph/project", json={"name": name, "seed_text": seed_text})
        return self._data(body).get("project_id") or body.get("project_id")

    def build_graph(self, project_id: str) -> str:
        body = self._json("POST", "/api/graph/build", json={"project_id": project_id})
        task_id = self._data(body).get("task_id") or body.get("task_id")
        graph_id = self._data(body).get("graph_id") or body.get("graph_id")
        if graph_id:
            return graph_id
        final = self._poll(f"/api/graph/task/{task_id}",
                           done=lambda b: self._data(b).get("status") in ("done", "completed", "success"),
                           what="graph build")
        return self._data(final).get("graph_id")

    def create_simulation(self, project_id: str, graph_id: str) -> str:
        body = self._json("POST", "/api/simulation/create", json={
            "project_id": project_id, "graph_id": graph_id,
            "enable_twitter": ENABLE_TWITTER, "enable_reddit": ENABLE_REDDIT,
        })
        return self._data(body).get("simulation_id") or body.get("simulation_id")

    def prepare(self, simulation_id: str, *, entity_types: list[str] | None = None) -> None:
        payload = {"simulation_id": simulation_id}
        if entity_types:   # scope entity→profile generation toward source/org entities
            payload["entity_types"] = entity_types
        self._json("POST", "/api/simulation/prepare", json=payload)
        self._poll("/api/simulation/prepare/status",
                   payload={"simulation_id": simulation_id},
                   done=lambda b: self._data(b).get("status") in ("ready", "done", "completed", "success"),
                   what="prepare")

    def start(self, simulation_id: str) -> None:
        # CONFIRM /api/simulation/start payload (rounds/steps may be passed here or set in prepare).
        self._json("POST", "/api/simulation/start", json={"simulation_id": simulation_id})
        self._poll(f"/api/simulation/{simulation_id}/run-status",
                   done=lambda b: self._data(b).get("status") in ("finished", "done", "completed", "success"),
                   what="simulation run")

    def generate_report(self, simulation_id: str) -> str:
        body = self._json("POST", "/api/report/generate", json={"simulation_id": simulation_id})
        report_id = self._data(body).get("report_id") or body.get("report_id")
        task_id = self._data(body).get("task_id") or body.get("task_id")
        self._poll("/api/report/generate/status",
                   payload={"task_id": task_id, "simulation_id": simulation_id},
                   done=lambda b: self._data(b).get("status") in ("done", "completed", "success"),
                   what="report")
        return report_id

    def get_sections(self, report_id: str) -> list[dict]:
        return self._data(self._json("GET", f"/api/report/{report_id}/sections")).get("sections", [])

    def get_graph(self, graph_id: str) -> dict:
        return self._data(self._json("GET", f"/api/graph/data/{graph_id}"))

    def interview(self, simulation_id: str, question: str) -> dict:
        # CONFIRM /api/simulation/interview(/all) payload + response shape on the box.
        return self._data(self._json("POST", "/api/simulation/interview/all",
                                     json={"simulation_id": simulation_id, "question": question}))

    # ── high-level: one full causal day-sim ───────────────────────────────────
    def run_day_sim(self, seed_text: str, question: str, *, name: str,
                    entity_types: list[str] | None = None) -> dict:
        """Run the whole pipeline for one decision day; return the raw artifacts.

        Returns ``{simulation_id, graph_id, report_id, sections, graph, votes}`` — the
        caller (sentisense.sim.extract / runner) turns these into the numeric feature +
        the stored graph/report. No interpretation here.
        """
        logger.info("MiroFish day-sim '{}' …", name)
        project_id = self.create_project(seed_text, name)
        graph_id = self.build_graph(project_id)
        simulation_id = self.create_simulation(project_id, graph_id)
        self.prepare(simulation_id, entity_types=entity_types)
        self.start(simulation_id)
        report_id = self.generate_report(simulation_id)
        out = {
            "simulation_id": simulation_id, "graph_id": graph_id, "report_id": report_id,
            "sections": self.get_sections(report_id),
            "graph": self.get_graph(graph_id),
            "votes": self.interview(simulation_id, question),
        }
        logger.info("  done: sim={} graph={} report={}", simulation_id, graph_id, report_id)
        return out
