"""
Shared state for the LangGraph RAG workflow.

Every node reads from / writes to this TypedDict.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """State that flows through all 8 workflow nodes."""

    # Inputs
    job_id: str
    question: str
    file_paths: list[str]

    # Prior Q&A in this session so follow-ups like "describe it" make sense
    chat_history: list[dict[str, str]]
    
    # Clarified question after resolving pronouns from history
    resolved_question: str

    # After ingest
    sections: list[dict[str, Any]]

    # After domain detection
    domain: str  # Legal | Insurance | HR | Finance

    # After chunk + index
    chunks: list[dict[str, Any]]
    num_indexed: int

    # After sub-query generation
    sub_queries: list[str]

    # After retrieve,  search results right after retrieval (before re-ranking)
    raw_hits: list[dict[str, Any]]

    # After re-rank, those results after re-ranking (sorted/filtered by relevance)
    ranked_hits: list[dict[str, Any]]

    # After generate
    analyst: dict[str, Any]
    auditor: dict[str, Any]

    # Self-correction bookkeeping
    correction_loop: int
    needs_correction: bool
    correction_reason: str

    # Final / error
    error: Optional[str]
    done: bool
