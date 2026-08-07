"""
Cross-encoder re-ranker using ms-marco-MiniLM-L-6-v2.

Takes (query, document) pairs and scores how relevant each document is.
Always keeps at least a few best hits so generation is never empty-handed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Return a cached CrossEncoder instance."""
    return CrossEncoder(settings.reranker_model)


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    min_score: float | None = None,
    top_k: int | None = None,
    min_keep: int | None = None,
) -> list[dict[str, Any]]:
    """
    Re-rank retrieval hits and drop low-relevance ones.

    Always keeps at least `min_keep` highest-scoring hits (even if below threshold)
    so the LLM still has evidence to work with.
    """
    if not hits:
        return []

    min_score = min_score if min_score is not None else settings.min_rerank_score
    top_k = top_k or settings.top_k
    min_keep = min_keep if min_keep is not None else settings.min_keep_hits

    model = get_reranker()
    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)

    scored = []
    for hit, score in zip(hits, scores):
        item = dict(hit)
        item["rerank_score"] = float(score)
        scored.append(item)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    filtered = [h for h in scored if h["rerank_score"] >= min_score]

    # Never return empty if we had candidates — keep best few
    if len(filtered) < min_keep:
        filtered = scored[: max(min_keep, min(top_k, len(scored)))]

    return filtered[:top_k]


def deduplicate_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate chunks (same source + chunk_id or identical text)."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []

    for hit in hits:
        key = f"{hit.get('source')}::{hit.get('chunk_id')}::{hit.get('text', '')[:80]}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)

    return unique
