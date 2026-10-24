"""Recherche de passages : BM25 lexical + score de couverture (sert à décider de s'abstenir).

L'interface `search(query, k) -> list[Hit]` est volontairement minimale : on peut la
remplacer par une recherche vectorielle (embeddings) ou hybride sans toucher au reste.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .ingest import Chunk
from .text import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # score BM25 brut
    coverage: float  # part (pondérée par l'IDF) des termes de la question présents dans le passage


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("Aucun passage à indexer")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.index_text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / len(chunks)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._n = len(chunks)
        self._df = df

    def idf(self, term: str) -> float:
        # IDF lissé ; un terme absent du corpus a l'IDF maximal (il est "inconnu")
        return math.log(1 + (self._n - self._df.get(term, 0) + 0.5) / (self._df.get(term, 0) + 0.5))

    def _bm25(self, q_terms: list[str], i: int) -> float:
        tf, dl = self._tf[i], self._len[i]
        score = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f:
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avg_len)
                score += self.idf(t) * f * (self.k1 + 1) / denom
        return score

    def _coverage(self, q_terms: list[str], i: int) -> float:
        total = sum(self.idf(t) for t in q_terms)
        if total == 0:
            return 0.0
        found = sum(self.idf(t) for t in q_terms if t in self._tf[i])
        return found / total

    name = "bm25"

    def rank_all(self, query: str) -> tuple[list[str], list[tuple[float, int]]]:
        """(termes de la requête, [(score, indice du passage)] triés par score décroissant)."""
        q_terms = sorted(set(tokenize(query)))
        scored = [(self._bm25(q_terms, i), i) for i in range(self._n)] if q_terms else []
        scored.sort(key=lambda x: (-x[0], x[1]))
        return q_terms, scored

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q_terms, scored = self.rank_all(query)
        if not q_terms:
            return []
        hits = []
        for score, i in scored[:k]:
            if score <= 0:
                break
            hits.append(Hit(self.chunks[i], score, self._coverage(q_terms, i)))
        return hits
