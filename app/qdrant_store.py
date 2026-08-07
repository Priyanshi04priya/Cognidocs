"""
Qdrant vector store helpers.

Each job gets its own collection (`job_{job_id}`) for tenant isolation.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.embeddings import embed_texts, embed_query


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client connected to the configured host."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def collection_name(job_id: str) -> str:
    """Per-job collection name for tenant isolation."""
    # Qdrant collection names: letters, digits, underscores, hyphens
    safe = job_id.replace("-", "_")
    return f"job_{safe}"


def create_collection(job_id: str) -> str:
    """Create (or recreate) a collection for this job."""
    client = get_qdrant_client()
    name = collection_name(job_id)

    # Drop existing collection if present (idempotent re-runs)
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    return name


def upsert_chunks(job_id: str, chunks: list[dict[str, Any]]) -> int:
    """
    Embed and upsert document chunks into the job's collection.

    Returns the number of points written.
    """
    if not chunks:
        return 0

    client = get_qdrant_client()
    name = collection_name(job_id)
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk.get("source", "unknown"),
                    "page": chunk.get("page", 0),
                    "chunk_id": chunk.get("chunk_id", 0),
                },
            )
        )

    # Upsert in batches of 64 to keep memory light
    batch_size = 64
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=name, points=points[i : i + batch_size])

    return len(points)


def search(
    job_id: str,
    query: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over a job's collection.

    Returns list of hits: {text, source, page, chunk_id, score}
    """
    top_k = top_k or settings.top_k
    client = get_qdrant_client()
    name = collection_name(job_id)
    vector = embed_query(query)

    results = client.search(
        collection_name=name,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )

    hits = []
    for r in results:
        payload = r.payload or {}
        hits.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "page": payload.get("page", 0),
                "chunk_id": payload.get("chunk_id", 0),
                "score": float(r.score),
            }
        )
    return hits


def delete_collection(job_id: str) -> None:
    """Remove a job's collection (cleanup)."""
    client = get_qdrant_client()
    name = collection_name(job_id)
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        client.delete_collection(name)
