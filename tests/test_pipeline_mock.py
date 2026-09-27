"""End-to-end smoke test of the offline pipeline on a small slice of the query set."""

import numpy as np

from r7b.cache import DiskCache
from r7b.clients import Client
from r7b.collect import collect
from r7b.data import load_queries
from r7b.features import handcrafted_matrix
from r7b.mock import MockBackend


def _slice():
    qs = load_queries()
    by_source = {}
    for q in qs:
        by_source.setdefault(q.source, []).append(q)
    return [q for group in by_source.values() for q in group[:6]]


def test_mock_collect_is_deterministic_and_cached(tmp_path):
    queries = _slice()
    backend = MockBackend(load_queries())
    first = collect(Client(backend, DiskCache(tmp_path)), queries, workers=2)
    client = Client(backend, DiskCache(tmp_path))
    second = collect(client, queries, workers=2)
    assert first == second
    assert client.stats["chat_calls"] == 0  # everything came from the cache
    for rec in first:
        assert rec["label"] in (0, 1)
        for role in ("cheap", "strong"):
            r = rec["responses"][role]
            assert 0 <= r["quality"] <= 1
            assert r["cost"] > 0
        assert rec["responses"]["strong"]["cost"] > rec["responses"]["cheap"]["cost"]
    assert all("judge" in r for r in first if r["source"] == "open")


def test_mock_embeddings_batch_and_cache(tmp_path):
    texts = [q.text for q in load_queries()[:200]]
    client = Client(MockBackend(), DiskCache(tmp_path))
    vecs, tokens = client.embed(texts)
    assert client.stats["embed_calls"] == 3  # 200 texts in batches of 96
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-4)
    assert all(t > 0 for t in tokens)
    client.embed(texts)
    assert client.stats["embed_calls"] == 3


def test_handcrafted_matrix_shape():
    m = handcrafted_matrix(["a", "b c"])
    assert m.shape[0] == 2
