# MiroFish agent-graph — UI data contract

The future SentiSense UI renders the **agent / knowledge graph** from each day's MiroFish
sim. The graph is persisted (`narrative_sim_graph`) and served by
`sentisense.sim.graph_api` — independent of whether the MiroFish service is running.

## JSON shape (normalized — stable regardless of MiroFish/Zep internals)
```json
{
  "sim_run_id": "sim_…",
  "date": "2024-03-15",
  "graph_id": "graph_…",
  "nodes": [ { "id": "n1", "type": "org|person|topic|agent|entity", "label": "…", "attrs": { } } ],
  "edges": [ { "src": "n1", "dst": "n2", "type": "mentions|relates|…", "weight": 1.0 } ],
  "meta":  { "n_nodes": 0, "n_edges": 0 }
}
```
`sentisense.sim.graph.normalize_graph` produces this from MiroFish's raw `/api/graph/data`,
so the UI never sees Zep/GraphRAG-version-specific field names.

## How the UI gets it
- **Latest (live panel):** `sentisense.sim.graph_api.latest_graph()` → the contract above.
- **By date:** `graph_for_date("2024-03-15")`.
- **Report (sidebar):** `report_for_date(date)` → `{report_id, report_md, sections}`.

Two front-end options:
1. **SentiSense UI** consumes the JSON above (e.g. via a thin Flask/FastAPI route wrapping
   `latest_graph`/`graph_for_date`), rendering nodes/edges with any graph lib (Cytoscape,
   D3, vis-network).
2. **MiroFish's own Vue UI** (`external/MiroFish/frontend`) already visualizes the live
   graph via `GET /api/graph/data/<graph_id>` — usable directly while the SentiSense UI is
   built; the normalized table is the durable, service-independent source of truth.

## Notes
- `attrs` carries the raw MiroFish node payload (persona, memory snippet, etc.) for
  rich tooltips — opaque to the contract, safe to ignore.
- Each sim's numeric verdict lives in `narrative_sim` (`dir_score`, `confidence`, …); join
  on `sim_date` to color/annotate the graph by the day's consensus.
