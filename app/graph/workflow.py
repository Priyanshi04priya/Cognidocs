"""
8-node stateful LangGraph workflow with self-correction branching.

Flow
----
ingest → detect_domain → chunk_and_index → generate_subqueries
      → retrieve → rerank_hits → generate_answers → self_correct
                                                      │
                              ┌───────────────────────┘
                              │ needs_correction?
                              ├─ yes → generate_subqueries (retry)
                              └─ no  → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    chunk_and_index,
    detect_domain,
    generate_answers,
    generate_subqueries,
    ingest_documents,
    rerank_hits,
    retrieve,
    self_correct,
)
from app.graph.state import GraphState


def _route_after_ingest(state: GraphState) -> str:
    if state.get("error") or state.get("done"):
        return END
    return "detect_domain"


def _route_after_index(state: GraphState) -> str:
    if state.get("error") or state.get("done"):
        return END
    return "generate_subqueries"


def _route_after_retrieve(state: GraphState) -> str:
    if state.get("error") or state.get("done"):
        return END
    return "rerank_hits"


def _route_after_generate(state: GraphState) -> str:
    if state.get("error") or state.get("done"):
        return END
    return "self_correct"


def _route_after_correction(state: GraphState) -> str:
    """Self-correction branch: retry sub-queries or finish."""
    if state.get("error"):
        return END
    if state.get("needs_correction"):
        return "generate_subqueries"
    return END


def build_workflow():
    """Compile and return the LangGraph app."""
    graph = StateGraph(GraphState)

    # Register all 8 nodes
    graph.add_node("ingest_documents", ingest_documents)
    graph.add_node("detect_domain", detect_domain)
    graph.add_node("chunk_and_index", chunk_and_index)
    graph.add_node("generate_subqueries", generate_subqueries)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank_hits", rerank_hits)
    graph.add_node("generate_answers", generate_answers)
    graph.add_node("self_correct", self_correct)

    # Entry
    graph.set_entry_point("ingest_documents")

    # Linear edges with early-exit on error
    graph.add_conditional_edges("ingest_documents", _route_after_ingest)
    graph.add_edge("detect_domain", "chunk_and_index")
    graph.add_conditional_edges("chunk_and_index", _route_after_index)
    graph.add_edge("generate_subqueries", "retrieve")
    graph.add_conditional_edges("retrieve", _route_after_retrieve)
    graph.add_edge("rerank_hits", "generate_answers")
    graph.add_conditional_edges("generate_answers", _route_after_generate)

    # Self-correction branching
    graph.add_conditional_edges("self_correct", _route_after_correction)

    return graph.compile()


# Singleton compiled graph (created once, reused by Celery workers)
rag_app = build_workflow()


def run_rag_pipeline(job_id: str, question: str, file_paths: list[str]) -> dict:
    """
    Convenience wrapper used by Celery tasks.

    Returns the final GraphState as a plain dict.
    """
    initial: GraphState = {
        "job_id": job_id,
        "question": question,
        "file_paths": file_paths,
        "correction_loop": 0,
        "needs_correction": False,
        "done": False,
    }
    return rag_app.invoke(initial)


def run_ingest_only(job_id: str, file_paths: list[str]) -> dict:
    """Upload path: ingest → detect domain → chunk + index into Qdrant."""
    state: GraphState = {
        "job_id": job_id,
        "question": "",
        "file_paths": file_paths,
        "correction_loop": 0,
        "needs_correction": False,
        "done": False,
    }
    state = ingest_documents(state)
    if state.get("error") or state.get("done"):
        return state
    state = detect_domain(state)
    state = chunk_and_index(state)
    return state


def run_ask_only(
    job_id: str,
    question: str,
    domain: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Ask path for an already-indexed session.
    Starts at sub-queries (skips ingest/index) so you can ask many questions
    on the same documents. Uses chat_history for follow-up understanding.
    """
    state: GraphState = {
        "job_id": job_id,
        "question": question,
        "file_paths": [],
        "domain": domain,
        "chat_history": chat_history or [],
        "correction_loop": 0,
        "needs_correction": False,
        "done": False,
    }

    while True:
        state = generate_subqueries(state)
        state = retrieve(state)
        if state.get("error") or state.get("done"):
            return state
        state = rerank_hits(state)
        state = generate_answers(state)
        if state.get("error") or state.get("done"):
            return state
        state = self_correct(state)
        if state.get("error") or not state.get("needs_correction"):
            return state
