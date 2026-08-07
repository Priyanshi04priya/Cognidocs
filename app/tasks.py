"""Celery tasks that run the LangGraph RAG pipeline off the web request."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.celery_app import celery_app
from app.graph.workflow import run_rag_pipeline

logger = logging.getLogger(__name__)


def execute_query_job(
    job_id: str,
    question: str,
    file_paths: list[str],
) -> dict[str, Any]:
    """
    Run the RAG pipeline and shape the API result.

    Used by both Celery workers and the local sync/thread fallback.
    """
    logger.info("Starting job %s", job_id)
    try:
        final = run_rag_pipeline(job_id=job_id, question=question, file_paths=file_paths)

        if final.get("error"):
            return {
                "job_id": job_id,
                "status": "failed",
                "domain": final.get("domain"),
                "question": question,
                "error": final["error"],
                "metadata": {
                    "correction_loop": final.get("correction_loop", 0),
                    "num_indexed": final.get("num_indexed", 0),
                },
            }

        return {
            "job_id": job_id,
            "status": "completed",
            "domain": final.get("domain"),
            "question": question,
            "analyst": final.get("analyst"),
            "auditor": final.get("auditor"),
            "error": None,
            "metadata": {
                "correction_loop": final.get("correction_loop", 0),
                "num_indexed": final.get("num_indexed", 0),
                "num_ranked_hits": len(final.get("ranked_hits") or []),
                "sub_queries": final.get("sub_queries") or [],
                "files": [Path(p).name for p in file_paths],
            },
        }
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        return {
            "job_id": job_id,
            "status": "failed",
            "question": question,
            "error": str(exc),
            "metadata": {},
        }


@celery_app.task(bind=True, name="app.tasks.run_query_job")
def run_query_job(
    self,
    job_id: str,
    question: str,
    file_paths: list[str],
) -> dict[str, Any]:
    """Celery wrapper around execute_query_job."""
    self.update_state(state="STARTED", meta={"job_id": job_id, "step": "pipeline"})
    return execute_query_job(job_id=job_id, question=question, file_paths=file_paths)
