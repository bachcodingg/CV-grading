"""Step 3 of the matcher: find the CV chunks most relevant to a requirement (PLAN.md §3 [5]).

Two retrievers share one interface, ``rank(query, chunks, k)`` returning
``(chunk_id, score)`` pairs with scores in [0, 1], best first:

- ``LexicalRetriever`` (always available, no models): idf-weighted share of
  the requirement's terms found in the chunk. Skill aliases are folded to
  their canonical name, so "K8s" and "Kubernetes" are the same term.
- ``EmbeddingRetriever`` (``matching`` extra): ``bge-small-en-v1.5`` on the
  CPU, cosine similarity.

An optional ``CrossEncoderReranker`` (``bge-reranker-base``, CPU) re-scores
the top-k and keeps the top-n. All models are forced to ``device="cpu"`` so
PyTorch never allocates GPU memory (the GPU budget belongs to the LLM).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Protocol

from app.config import MatchingConfig
from app.matching.schemas import Chunk
from app.matching.skills import find_skills

log = logging.getLogger(__name__)


class Retriever(Protocol):
    name: str

    def rank(self, query: str, chunks: list[Chunk], k: int) -> list[tuple[int, float]]: ...


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[int, float]]: ...


# ------------------------------------------------------------------ lexical

_STOP = set("""
a an and are as at be been but by can for from has have in into is it its of on or our such that the their this
to was we were will with you your who what which within across using use used via per etc also well able ability
strong solid good great excellent deep proven experience experienced knowledge skills skill understanding working
work years year plus required preferred ideally must nice bonus least minimum including e g ie eg professional
relevant practical hands-on
""".split())
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#.]*[A-Za-z0-9+#]|[A-Za-z]")


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "ers", "er", "ies", "ed", "es", "ly", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def terms(text: str) -> set[str]:
    out = {f"skill:{m.canonical.lower()}" for m in find_skills(text)}
    for m in _WORD.finditer(text):
        w = m.group().lower().rstrip(".")
        if w not in _STOP and len(w) > 1:
            out.add(_stem(w))
    return out


class LexicalRetriever:
    name = "lexical"

    def __init__(self) -> None:
        self._cache_key: tuple | None = None
        self._docs: dict[int, set[str]] = {}
        self._idf: dict[str, float] = {}

    def _prepare(self, chunks: list[Chunk]) -> None:
        key = tuple((c.id, c.search_text) for c in chunks)
        if key == self._cache_key:
            return
        self._cache_key = key
        self._docs = {c.id: terms(c.search_text) for c in chunks}
        n = len(chunks)
        df: dict[str, int] = {}
        for ts in self._docs.values():
            for t in ts:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + n / d) for t, d in df.items()}
        self._unseen = math.log(1 + n) if n else 1.0

    def rank(self, query: str, chunks: list[Chunk], k: int) -> list[tuple[int, float]]:
        if not chunks:
            return []
        self._prepare(chunks)
        q = terms(query)
        if not q:
            return []
        total = sum(self._idf.get(t, self._unseen) for t in q)
        scored = []
        for c in chunks:
            hit = sum(self._idf.get(t, self._unseen) for t in q & self._docs[c.id])
            if hit:
                scored.append((c.id, round(hit / total, 4)))
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:k]


# ------------------------------------------------------------------ models (CPU)


class EmbeddingRetriever:
    """Cosine similarity of sentence embeddings. ``model`` needs ``encode(list[str], normalize_embeddings=True)``."""

    name = "embeddings"

    def __init__(self, model, query_prefix: str = "") -> None:
        self.model = model
        self.query_prefix = query_prefix
        self._cache_key: tuple | None = None
        self._vectors = None

    def _encode(self, texts: list[str]):
        import numpy as np

        vecs = np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def rank(self, query: str, chunks: list[Chunk], k: int) -> list[tuple[int, float]]:
        if not chunks:
            return []
        key = tuple((c.id, c.search_text) for c in chunks)
        if key != self._cache_key:
            self._vectors = self._encode([c.search_text for c in chunks])
            self._cache_key = key
        q = self._encode([self.query_prefix + query])[0]
        sims = self._vectors @ q
        order = sorted(range(len(chunks)), key=lambda i: (-float(sims[i]), chunks[i].id))[:k]
        return [(chunks[i].id, round(max(float(sims[i]), 0.0), 4)) for i in order]


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly. ``model`` needs ``predict(list[tuple[str, str]])``."""

    name = "reranker"

    def __init__(self, model) -> None:
        self.model = model

    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[int, float]]:
        if not chunks:
            return []
        raw = [float(s) for s in self.model.predict([(query, c.search_text) for c in chunks],
                                                     show_progress_bar=False)]
        # bge-reranker returns sigmoid scores; a model returning logits is squashed to [0, 1].
        if any(s < 0 or s > 1 for s in raw):
            raw = [1 / (1 + math.exp(-s)) for s in raw]
        scored = sorted(zip((c.id for c in chunks), raw), key=lambda p: (-p[1], p[0]))
        return [(cid, round(s, 4)) for cid, s in scored]


def load_embedding_retriever(cfg: MatchingConfig) -> EmbeddingRetriever:
    """Raises ImportError if the ``matching`` extra is missing."""
    from sentence_transformers import SentenceTransformer

    return EmbeddingRetriever(SentenceTransformer(cfg.embedding_model, device="cpu"), cfg.query_prefix)


def load_reranker(cfg: MatchingConfig) -> CrossEncoderReranker:
    from sentence_transformers import CrossEncoder

    return CrossEncoderReranker(CrossEncoder(cfg.reranker_model, device="cpu"))
