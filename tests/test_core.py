"""Lightweight unit tests that don't need Redis / Qdrant / OpenAI."""

from app.chunker import chunk_text, chunk_sections
from app.document_loader import load_document
from app.graph.nodes import detect_domain, DOMAIN_KEYWORDS
from app.graph.state import GraphState
from app.reranker import deduplicate_hits


def test_chunk_overlap():
    text = "A" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) >= 3
    assert len(chunks[0]) == 1000
    # Overlap check on hard-split oversized units
    assert chunks[0][-200:] == chunks[1][:200]


def test_chunk_prefers_paragraphs():
    text = "First paragraph about leave policy.\n\nSecond paragraph about termination notice and benefits."
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    assert "leave policy" in chunks[0]
    assert "termination" in chunks[0]


def test_rerank_keeps_minimum_hits():
    from app.reranker import rerank

    hits = [
        {"text": "unrelated sports news", "source": "a.pdf", "chunk_id": 0},
        {"text": "employee leave entitlement is twenty days", "source": "b.pdf", "chunk_id": 1},
        {"text": "random cooking recipe", "source": "c.pdf", "chunk_id": 2},
    ]
    # High threshold would wipe everything — min_keep must still return hits
    ranked = rerank("leave entitlement", hits, min_score=99.0, top_k=5, min_keep=2)
    assert len(ranked) >= 2
    assert ranked[0]["text"]  # best should be leave-related ideally


def test_chunk_sections_adds_ids():
    sections = [{"text": "Hello world " * 200, "source": "a.pdf", "page": 1}]
    chunks = chunk_sections(sections)
    assert chunks
    assert chunks[0]["chunk_id"] == 0
    assert chunks[0]["source"] == "a.pdf"


def test_load_eml():
    sections = load_document("sample_docs/hr_policy.eml")
    assert sections
    assert "leave" in sections[0]["text"].lower()
    assert sections[0]["source"] == "hr_policy.eml"


def test_detect_domain_hr():
    state: GraphState = {
        "job_id": "test",
        "question": "What is the leave policy?",
        "file_paths": [],
        "sections": [
            {
                "text": " ".join(DOMAIN_KEYWORDS["HR"]) + " employee payroll benefits leave",
                "source": "hr.eml",
                "page": 1,
            }
        ],
    }
    out = detect_domain(state)
    assert out["domain"] == "HR"


def test_deduplicate_hits():
    hits = [
        {"text": "same", "source": "a.pdf", "chunk_id": 1},
        {"text": "same", "source": "a.pdf", "chunk_id": 1},
        {"text": "other", "source": "b.pdf", "chunk_id": 2},
    ]
    unique = deduplicate_hits(hits)
    assert len(unique) == 2
