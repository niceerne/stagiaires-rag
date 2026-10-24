"""Recherche hybride : BM25 (mots-clés) + embeddings (sens), fusionnés par RRF.

BM25 est excellent quand la question reprend les mots du document ; les embeddings retrouvent les
synonymes (« bus » ~ « transport en commun »). La fusion RRF (Reciprocal Rank Fusion) combine les deux
classements sans avoir à comparer des scores d'échelles différentes.
Les embeddings viennent d'Ollama (`ollama pull bge-m3`), donc gratuits et locaux.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path

from .retriever import BM25Retriever, Hit


class OllamaEmbedder:
    def __init__(self, host: str | None = None, model: str | None = None, post=None, timeout: int = 300):
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = "http://" + host
        self.host = host.rstrip("/")
        self.model = model or os.environ.get("RAG_EMBED_MODEL", "bge-m3")
        self.timeout = timeout
        self._post = post or self._http_post

    def _http_post(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post(f"{self.host}/api/embed", {"model": self.model, "input": texts})
        vecs = data["embeddings"]
        if len(vecs) != len(texts):
            raise ValueError("nombre d'embeddings inattendu")
        return [_normalize(v) for v in vecs]


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class HybridRetriever:
    """Même interface que BM25Retriever (`search`, `chunks`). Se replie sur BM25 si l'embedding échoue."""

    def __init__(self, bm25: BM25Retriever, embedder, cache_path: str | Path | None = None,
                 rrf_k: int = 60, pool: int = 10):
        self.bm25, self.embedder = bm25, embedder
        self.chunks = bm25.chunks
        self.rrf_k, self.pool = rrf_k, pool
        self.name = f"hybrid:{embedder.model}"
        self.fallbacks = 0
        self._vecs = self._embed_chunks(cache_path)

    def _embed_chunks(self, cache_path) -> list[list[float]]:
        path = Path(cache_path) if cache_path else None
        cache: dict[str, list[float]] = {}
        if path and path.exists():
            try:
                cache = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cache = {}
        keys = [hashlib.sha256(f"{self.embedder.model}\n{c.index_text}".encode()).hexdigest() for c in self.chunks]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            for i, vec in zip(missing, self.embedder.embed([self.chunks[i].index_text for i in missing])):
                cache[keys[i]] = vec
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cache), encoding="utf-8")
        return [cache[k] for k in keys]

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q_terms, ranked = self.bm25.rank_all(query)
        if not q_terms:
            return []
        lexical = [i for s, i in ranked if s > 0][: self.pool]
        try:
            qv = self.embedder.embed([query])[0]
            dense = [i for _, i in sorted(((_dot(qv, v), i) for i, v in enumerate(self._vecs)), reverse=True)]
            dense = dense[: self.pool]
        except Exception:  # Ollama éteint, modèle absent... -> BM25 seul
            self.fallbacks += 1
            dense = []
        fused: dict[int, float] = {}
        for ranking in (lexical, dense):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [Hit(self.chunks[i], score, self.bm25._coverage(q_terms, i)) for i, score in top]
