"""Normalize MiroFish's /api/graph/data into a stable UI contract.

MiroFish's graph comes from Zep/GraphRAG; the exact field names vary by version, so we
map defensively into the contract the future SentiSense UI consumes:

    {"nodes": [{"id", "type", "label", "attrs"}],
     "edges": [{"src", "dst", "type", "weight"}],
     "meta":  {"n_nodes", "n_edges"}}
"""

from __future__ import annotations


def _first_list(d, keys) -> list:
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for k in keys:
            v = d.get(k)
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                inner = _first_list(v, keys)
                if inner:
                    return inner
    return []


def _get(d, *keys, default=None):
    if isinstance(d, dict):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
    return default


def normalize_graph(raw) -> dict:
    """Map a raw MiroFish graph payload to the UI node/edge contract (defensive)."""
    nodes_raw = _first_list(raw, ("nodes", "entities", "vertices"))
    edges_raw = _first_list(raw, ("edges", "relations", "links", "relationships", "triples"))

    nodes = []
    for i, n in enumerate(nodes_raw):
        nid = (_get(n, "id", "uuid", "name", "label") if isinstance(n, dict) else n) or f"node_{i}"
        nodes.append({
            "id": str(nid),
            "type": str(_get(n, "type", "category", "entity_type", default="entity")),
            "label": str(_get(n, "name", "label", "title", "summary", default=nid)),
            "attrs": n if isinstance(n, dict) else {},
        })

    edges = []
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        try:
            weight = float(_get(e, "weight", "score", "strength", default=1.0))
        except (TypeError, ValueError):
            weight = 1.0
        edges.append({
            "src": str(_get(e, "src", "source", "from", "start", "subject", "head", default="")),
            "dst": str(_get(e, "dst", "target", "to", "end", "object", "tail", default="")),
            "type": str(_get(e, "type", "relation", "label", "predicate", default="rel")),
            "weight": weight,
        })

    return {"nodes": nodes, "edges": edges,
            "meta": {"n_nodes": len(nodes), "n_edges": len(edges)}}
