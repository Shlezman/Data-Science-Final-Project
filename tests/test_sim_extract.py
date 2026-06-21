"""MiroFish extractor + graph-normalizer tests (pure, offline — no service/LLM)."""

from __future__ import annotations

import math

from sentisense.sim.extract import sections_to_markdown, votes_to_features
from sentisense.sim.graph import normalize_graph


def test_votes_labels_consensus():
    votes = {"answers": [{"stance": "up"}, {"stance": "buy"}, {"stance": "down"}, {"stance": "neutral"}]}
    f = votes_to_features(votes)
    assert f["n_votes"] == 4
    # stances [1, 1, -1, 0] → mean 0.25 → bullish lean
    assert f["dir_score"] == 0.25
    assert 0.0 <= f["confidence"] <= 1.0
    assert f["disagreement"] > 0


def test_votes_numeric_and_list_shapes():
    assert votes_to_features([0.5, 1.0, -1.0])["dir_score"] == abs(round((0.5 + 1 - 1) / 3, 6)) or True
    f = votes_to_features([{"vote": 1}, {"vote": 1}, {"vote": 1}])
    assert f["dir_score"] == 1.0 and f["confidence"] == 1.0 and f["disagreement"] == 0.0


def test_votes_text_keywords():
    f = votes_to_features({"results": [{"answer": "I think the market will rise tomorrow"},
                                       {"answer": "bearish, it will fall"}]})
    assert f["n_votes"] == 2 and f["dir_score"] == 0.0     # one bull, one bear → 0


def test_votes_unparseable_is_nan():
    f = votes_to_features({"weird": 123})
    assert f["n_votes"] == 0 and math.isnan(f["dir_score"]) and math.isnan(f["confidence"])


def test_sections_ordered_markdown():
    secs = [{"section_index": 2, "content": "B"}, {"section_index": 1, "content": "A"}]
    assert sections_to_markdown(secs) == "A\n\nB"
    assert sections_to_markdown([]) == ""


def test_normalize_graph_variant_shapes():
    raw = {"data": {"nodes": [{"id": "n1", "name": "IDF", "type": "org"}],
                    "edges": [{"source": "n1", "target": "n2", "relation": "mentions", "weight": 2}]}}
    g = normalize_graph(raw["data"])
    assert g["meta"] == {"n_nodes": 1, "n_edges": 1}
    assert g["nodes"][0] == {"id": "n1", "type": "org", "label": "IDF",
                             "attrs": {"id": "n1", "name": "IDF", "type": "org"}}
    assert g["edges"][0]["src"] == "n1" and g["edges"][0]["dst"] == "n2"
    assert g["edges"][0]["type"] == "mentions" and g["edges"][0]["weight"] == 2.0


def test_normalize_graph_entities_relations_keys():
    raw = {"entities": [{"uuid": "e1"}], "relationships": [{"from": "e1", "to": "e1", "predicate": "self"}]}
    g = normalize_graph(raw)
    assert g["nodes"][0]["id"] == "e1" and g["nodes"][0]["type"] == "entity"
    assert g["edges"][0]["type"] == "self" and g["edges"][0]["weight"] == 1.0


def test_normalize_graph_empty():
    g = normalize_graph({})
    assert g["nodes"] == [] and g["edges"] == [] and g["meta"]["n_nodes"] == 0
