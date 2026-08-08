"""
Eight LangGraph nodes for the RAG pipeline.

Node map
--------
1. ingest_documents   – load PDF / DOCX / EML
2. detect_domain      – classify Legal / Insurance / HR / Finance
3. chunk_and_index    – 1000-char chunks → per-job Qdrant collection
4. generate_subqueries – create 3–5 search sub-queries
5. retrieve           – concurrent top-5 semantic search (asyncio)
6. rerank_hits        – CrossEncoder re-rank + dedupe
7. generate_answers   – dual-persona Analyst + Auditor (GPT-4.1 mini)
8. self_correct       – quality check; branch back if needed
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.chunker import chunk_sections
from app.config import settings, has_openai_key
from app.document_loader import load_documents
from app.graph.state import GraphState
from app.qdrant_store import create_collection, search, upsert_chunks
from app.reranker import deduplicate_hits, rerank

logger = logging.getLogger(__name__)

DOMAINS = ["Legal", "Insurance", "HR", "Finance"]

#client
def _llm() -> ChatOpenAI:
    """Create a ChatOpenAI client (GPT-4.1 mini by default). Low temp = more faithful."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key or None,
        temperature=0,
    )


# ===================================================================
# 1. Ingest documents
# ===================================================================

def ingest_documents(state: GraphState) -> GraphState:
    """Load PDF, DOCX, and EML files into plain-text sections."""
    #output me state update hoga 
    #log me update karo jb bhi ingest ho rha hai
    logger.info("[%s] ingest_documents", state.get("job_id"))
    try:
        #file_paths uploaded files(more than one file can be there) ka path hoga, load_documents function me pass karenge
        sections = load_documents(state["file_paths"])
        if not sections:
            #If extraction returned nothing (empty list) then copy old state and we set an error message and mark the state as done
            #done:True→stop the pipeline
            return {**state, "error": "No text could be extracted from the uploaded files.", "done": True}
        return {**state, "sections": sections, "error": None}
    except Exception as exc:
        return {**state, "error": f"Ingest failed: {exc}", "done": True}


# ===================================================================
# 2. Detect domain
# ===================================================================

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Legal": [
        "contract", "clause", "liability", "jurisdiction", "plaintiff",
        "defendant", "indemnif", "agreement", "whereas", "statute", "court",
    ],
    "Insurance": [
        "premium", "policyholder", "coverage", "deductible", "claim",
        "underwrit", "insured", "beneficiary", "actuarial", "rider",
    ],
    "HR": [
        "employee", "payroll", "onboarding", "benefits", "leave",
        "performance review", "hiring", "termination", "hr policy", "workforce",
    ],
    "Finance": [
        "revenue", "balance sheet", "invoice", "fiscal", "audit",
        "equity", "cash flow", "budget", "profit", "accounts payable",
    ],
}


def detect_domain(state: GraphState) -> GraphState:
    """
    Auto-detect document domain using keyword scoring + optional LLM tie-break.
    Falls back to keyword-only if no API key is set.
    """
    logger.info("[%s] detect_domain", state.get("job_id"))
    sections = state.get("sections") or []

    """First 5 sections only
       First 500 characters of each
       Join into one string
       Lowercase it
       Why? Fast check — no need to scan the whole document."""
    sample = " ".join(s["text"][:500] for s in sections[:5]).lower()

    scores = {
        domain: sum(1 for kw in kws if kw in sample)
        for domain, kws in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    best_score = scores[best]

    #phle khud se check karenge why bcz its cheap and fast, then if score is low or tied we will ask LLM to classify the domain
    #as we are calling llm so we will check if openai key is set or not, if not then we will use keyword based classification
    if best_score < 2 and has_openai_key():
        try:
            llm = _llm()
            prompt = (
                "Classify the following document excerpt into exactly one domain: "
                "Legal, Insurance, HR, or Finance.\n"
                "Reply with ONLY the domain name.\n\n"
                f"Excerpt:\n{sample[:1500]}"
            )
            reply = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            for d in DOMAINS:
                if d.lower() in reply.lower():
                    best = d
                    break
        except Exception as exc:
            logger.warning("LLM domain detection failed, using keywords: %s", exc)

    return {**state, "domain": best}


# ===================================================================
# 3. Chunk and index into Qdrant
# ===================================================================

def chunk_and_index(state: GraphState) -> GraphState:
    """Chunk text (1000 / 200 overlap) and upsert into a per-job Qdrant collection."""
    logger.info("[%s] chunk_and_index", state.get("job_id"))
    try:
        chunks = chunk_sections(state["sections"])
        if not chunks:
            return {**state, "error": "Chunking produced no text chunks.", "done": True}

        create_collection(state["job_id"])
        n = upsert_chunks(state["job_id"], chunks)
        return {**state, "chunks": chunks, "num_indexed": n}
    except Exception as exc:
        return {**state, "error": f"Indexing failed: {exc}", "done": True}


# ======================================================================
# 4. Generate sub-queries   -------3 functions are under this section
# ======================================================================

#This function turns past Q&A turns into one text string the LLM can read as chat history.
def _format_history(history: list[dict[str, str]], limit: int = 4) -> str:
    if not history: #if history is empty / missing → return empty string
        return ""
    lines = [] #to store history in one string
    for turn in history[-limit:]:  #Loop over only the last limit turns.
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            lines.append(f"User: {q}")
        if a:
            lines.append(f"Assistant: {a[:500]}")
    return "\n".join(lines)



"""New question comes in
   No history? → return as-is
   Build cheap fallback
   (if short + vague words → add "regarding: last question")
   No OpenAI key? → return fallback
    ↓
   Ask LLM to rewrite clearly using history
    ↓
   Success → return rewritten question
   Fail/empty → return fallback"""

def _resolve_followup_question(question: str, history: list[dict[str, str]], domain: str) -> str:
    """
    Turn vague follow-ups ("describe it whole") into a clear standalone question
    using earlier conversation turns.
    """
    if not history:
        return question

    # Cheap heuristic fallback without LLM
    fallback = question
    last_q = (history[-1].get("question") or "").strip()
    if last_q and len(question.split()) <= 12:
        low = question.lower()
        if any(w in low for w in ("it", "this", "that", "them", "above", "same", "whole", "more", "detail")):
            fallback = f"{question} (regarding: {last_q})"

    if not has_openai_key():
        return fallback

    try:
        llm = _llm()
        hist = _format_history(history)
        prompt = (
            "Rewrite the latest user message into ONE clear standalone question "
            "for searching a document. Resolve pronouns like it/this/that using the chat history. "
            "Keep the user's intent. Return ONLY the rewritten question text.\n\n"
            f"Domain: {domain}\n"
            f"Chat history:\n{hist}\n\n"
            f"Latest user message: {question}"
        )
        rewritten = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        rewritten = rewritten.strip('"').strip("'")
        return rewritten or fallback
    except Exception as exc:
        logger.warning("Follow-up rewrite failed: %s", exc)
        return fallback


def generate_subqueries(state: GraphState) -> GraphState:
    """Create 3–5 focused sub-queries from the user question (with chat context)."""
    logger.info("[%s] generate_subqueries (loop=%s)", state.get("job_id"), state.get("correction_loop", 0))

    raw_question = state["question"] #Exact latest user message (may be vague, like "describe it").
    domain = state.get("domain", "General")#Document domain, or "General" if missing.

    #correction_hint = feedback from the last failed attempt, used to improve the next search.
    #updated through self_correct node, then passed back here to diversify the sub-queries.
    #first time through → empty string, then filled in if the Analyst answer was weak.

    correction_hint = state.get("correction_reason", "")
    history = state.get("chat_history") or [] #Past Q&A turns. If missing → empty list

    resolved = state.get("resolved_question") or _resolve_followup_question(
        raw_question, history, domain  #If resolved_question already exists in state → reuse it
        #Else call _resolve_followup_question to turn vague follow-ups into a clear question using history
    )

    # Default fallback sub-queries (work without an API key)
    # Include raw + resolved + keyword-ish variants for better recall
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", resolved) if t.lower() not in {
        "the", "and", "for", "what", "how", "who", "when", "where", "with", "from",
        "this", "that", "have", "describe", "explain", "brief", "whole", "detail",
        "want", "you", "please",
        # pulls word-like tokens from resolved (letters/numbers, length ≥ 3) Drops common useless words: the, and, what, describe, please, etc.
        # Keeps meaningful words only
    }]
    keyword_query = " ".join(tokens[:8]) if tokens else resolved


    """5 backup search angles if LLM is unavailable/fails:
    Full clear question
    Keyword-only version
    Domain-tagged question
    “definition/overview” style,  Asks for a broad explanation: what it is, main features, overview. 
    “key details” style, Asks for important specifics (numbers, conditions, exceptions), not just a definition.
    Why 5 angles?
    One query can miss relevant chunks. Multiple phrasings increase the chance of hitting the right passages.
    """
    fallback = [
        resolved,
        keyword_query,
        f"{domain}: {resolved}",
        f"definition features overview of {keyword_query}",
        f"Key details about {keyword_query}",
    ]

    if not has_openai_key():
        return {**state, "resolved_question": resolved, "sub_queries": fallback[:4]}

    try:
        llm = _llm()
        system = (
            "You generate search sub-queries for a document RAG system. "
            "Given a user question and document domain, produce 3 to 5 short, "
            "diverse search queries that would help retrieve relevant clauses. "
            "Return a JSON array of strings only. No markdown."
        )
        user = f"Domain: {domain}\nQuestion: {resolved}"
        if history:
            user += f"\nRecent chat context:\n{_format_history(history, limit=2)}"
        if correction_hint:
            user += f"\nPrevious search was weak because: {correction_hint}. Diversify the queries."

        reply = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        ).content.strip()

        cleaned = reply.strip("`").removeprefix("json").strip()
        queries = json.loads(cleaned)

        #isinstance(value, type) checks: “Is this value this type?”
        if not isinstance(queries, list) or not queries:
            queries = fallback
        queries = [str(q) for q in queries][:5]
        if len(queries) < 3:
            queries = (queries + fallback)[:4]
    except Exception as exc:
        logger.warning("Sub-query generation failed, using fallback: %s", exc)
        queries = fallback[:4]

    return {**state, "resolved_question": resolved, "sub_queries": queries}


# ===================================================================
# 5. Concurrent retrieve
# ===================================================================

async def _search_one(job_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
    """Run one Qdrant search for one sub-query in a thread so we don't block the event loop."""
    return await asyncio.to_thread(search, job_id, query, top_k)


async def _retrieve_all(job_id: str, queries: list[str], top_k: int) -> list[dict[str, Any]]:
    """Instead of searching one-by-one (slow), search all in parallel (faster), then combine."""
    tasks = [_search_one(job_id, q, top_k) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True) #Run all tasks concurrently. return_exceptions=True means: if one fails, don’t crash

    hits: list[dict[str, Any]] = [] #store results of subquery searches
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Search task failed: %s", result)
            continue
        hits.extend(result)
    return hits


def retrieve(state: GraphState) -> GraphState:
    """Execute semantic search over each sub-query concurrently (retrieve_k each)."""
    logger.info("[%s] retrieve (%d sub-queries)", state.get("job_id"), len(state.get("sub_queries") or []))
    try:
        hits = asyncio.run(
            _retrieve_all(
                state["job_id"],
                state["sub_queries"],
                getattr(settings, "retrieve_k", settings.top_k),
            )
        )
        return {**state, "raw_hits": hits} #Success → save all found chunks as raw_hits.
    except Exception as exc:
        return {**state, "error": f"Retrieval failed: {exc}", "done": True} #Something broke → save error and stop pipeline.


# ===================================================================
# 6. Re-rank + dedupe
# ===================================================================

def rerank_hits(state: GraphState) -> GraphState:
    """Deduplicate hits, then re-rank with CrossEncoder and drop weak ones."""
    logger.info("[%s] rerank_hits", state.get("job_id"))
    unique = deduplicate_hits(state.get("raw_hits") or [])

    # Prefer the resolved (context-aware) question for re-ranking
    query = state.get("resolved_question") or state["question"]
    #Uses a CrossEncoder model (ms-marco-MiniLM...)
    #It uses a trained neural model (CrossEncoder on MS MARCO) that learned from many examples
    ranked = rerank(query, unique)
    return {**state, "ranked_hits": ranked}


# ===================================================================
# 7. Generate dual-persona answers
# ===================================================================

def _build_context(hits: list[dict[str, Any]]) -> str:
    """Format ranked hits into a numbered context block for the LLM."""
    lines = []
    for i, h in enumerate(hits, start=1):
        score = h.get("rerank_score", h.get("score", 0))
        lines.append(
            f"[{i}] source={h.get('source')} page={h.get('page')} score={score:.3f}\n"
            f"{h.get('text', '')}"
        )
    return "\n\n".join(lines)


def _parse_persona_json(raw: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse LLM JSON into a PersonaAnswer-like dict with safe defaults."""
    try:
        cleaned = raw.strip().strip("`").removeprefix("json").strip()
        data = json.loads(cleaned)
    except Exception:
        data = {
            "answer": raw.strip(),
            "confidence": 0.4,
            "citations": [],
            "justification": ["Model returned non-JSON; raw text used as answer."],
        }

    # Attach real clause text when citation indices are given
    citations = []
    for c in data.get("citations") or []:
        if isinstance(c, dict):
            citations.append(
                {
                    "source": c.get("source", "unknown"),
                    "clause": c.get("clause", ""),
                    "score": float(c.get("score", 0.0)),
                }
            )
        elif isinstance(c, int) and 1 <= c <= len(hits):
            h = hits[c - 1]
            citations.append(
                {
                    "source": h.get("source", "unknown"),
                    "clause": h.get("text", "")[:400],
                    "score": float(h.get("rerank_score", h.get("score", 0))),
                }
            )

    return {
        "answer": data.get("answer", ""),
        "confidence": float(data.get("confidence", 0.5)),
        "citations": citations,
        "justification": list(data.get("justification") or []),
    }


def _persona_prompt(
    role: str,
    domain: str,
    question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
    resolved_question: str | None = None,
) -> list:
    """Build system + user messages for Analyst or Auditor."""
    shared_rules = (
        "STRICT GROUNDING RULES:\n"
        "1. Use ONLY facts that appear in the Context passages below.\n"
        "2. Do NOT use outside/world knowledge. If the docs don't say it, don't invent it.\n"
        "3. Prefer short quotes from the context and name the source file.\n"
        "4. If context is partial, answer what IS supported and clearly say what is missing.\n"
        "5. Follow-ups like 'describe it' must use chat history to resolve the topic, "
        "then answer from the documents — never claim the question is unclear if history names it.\n"
        "6. Return STRICT JSON with keys: answer (string), confidence (0-1 float), "
        "citations (list of {source, clause, score}), justification (list of steps)."
    )

    if role == "Analyst":
        system = (
            f"You are a careful {domain} Analyst reading uploaded documents.\n"
            "Give a clear, accurate answer grounded in the passages.\n"
            f"{shared_rules}"
        )
    else:
        system = (
            f"You are a skeptical {domain} Auditor reviewing the same documents.\n"
            "Confirm what the docs support, flag weak/missing evidence, and avoid speculation.\n"
            f"{shared_rules}"
        )

    hist_block = _format_history(history or [], limit=4) or "(none)"
    user = (
        f"Domain: {domain}\n"
        f"Latest user question: {question}\n"
        f"Resolved question: {resolved_question or question}\n\n"
        f"Chat history:\n{hist_block}\n\n"
        f"Context passages (ONLY source of truth):\n{context}\n\n"
        "Write the answer using only those passages. JSON only."
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _demo_citations(hits: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "source": h.get("source", "unknown"),
            "clause": (h.get("text") or "")[:500],
            "score": float(h.get("rerank_score", h.get("score", 0))),
        }
        for h in hits[:limit]
    ]


def _extractive_demo_answers(
    question: str,
    domain: str,
    hits: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build readable Analyst/Auditor answers from retrieved text when no OpenAI key is set.

    This is NOT an LLM answer — it quotes the best matching passages so beginners
    can still see what the RAG pipeline found.
    """
    citations = _demo_citations(hits)

    if not hits:
        empty = {
            "answer": (
                "No relevant passages were found in your documents for this question. "
                "Try a simpler question, or upload a document that actually contains the topic."
            ),
            "confidence": 0.1,
            "citations": [],
            "justification": [
                "Ran retrieval + re-ranking.",
                "Nothing scored high enough to keep.",
                "Tip: set OPENAI_API_KEY in .env for full GPT explanations.",
            ],
        }
        return empty, {
            **empty,
            "answer": (
                "Auditor check: retrieval returned no usable evidence, so no claim can be verified."
            ),
        }

    # Quote the top passages clearly
    quoted_parts = []
    for i, h in enumerate(hits[:3], start=1):
        text = (h.get("text") or "").strip()
        source = h.get("source", "unknown")
        page = h.get("page", "?")
        quoted_parts.append(f"Passage {i} (from {source}, page {page}):\n{text}")

    evidence = "\n\n".join(quoted_parts)

    analyst = {
        "answer": (
            f"**Demo mode (no OpenAI API key)** — domain detected: **{domain}**\n\n"
            f"**Your question:** {question}\n\n"
            f"Here are the most relevant excerpts found in your documents:\n\n"
            f"{evidence}\n\n"
            "—\n"
            "This is extractive RAG output (quoted text only). "
            "Add `OPENAI_API_KEY` in `.env` to get a real Analyst explanation written by GPT-4.1 mini."
        ),
        "confidence": min(0.55, 0.3 + 0.1 * len(hits)),
        "citations": citations,
        "justification": [
            f"Detected document domain as {domain}.",
            f"Ran semantic search and kept {len(hits)} re-ranked passage(s).",
            "Quoted the top passages below because no OpenAI key is configured.",
            "Set OPENAI_API_KEY for dual-persona GPT answers with reasoning.",
        ],
    }

    auditor = {
        "answer": (
            f"**Demo mode Auditor review**\n\n"
            f"Evidence was retrieved for: “{question}”.\n\n"
            f"I found {len(hits)} passage(s). Top source: "
            f"**{hits[0].get('source', 'unknown')}** "
            f"(re-rank score {float(hits[0].get('rerank_score', 0)):.3f}).\n\n"
            "Caveats without an LLM:\n"
            "- These quotes are raw excerpts, not a verified legal/HR interpretation.\n"
            "- Confirm the full clause in the original file before relying on it.\n"
            "- Add OPENAI_API_KEY for a proper Auditor critique with gap analysis."
        ),
        "confidence": min(0.5, 0.25 + 0.1 * len(hits)),
        "citations": citations,
        "justification": [
            "Checked that retrieval returned at least one passage.",
            "Flagged that extractive quotes are not a substitute for expert review.",
            "Recommend enabling OpenAI for full Auditor persona output.",
        ],
    }
    return analyst, auditor


def generate_answers(state: GraphState) -> GraphState:
    """Produce Analyst + Auditor answers with confidence, citations, justification."""
    logger.info("[%s] generate_answers", state.get("job_id"))
    hits = state.get("ranked_hits") or []
    domain = state.get("domain", "General")
    question = state["question"]
    resolved = state.get("resolved_question") or question
    history = state.get("chat_history") or []
    context = _build_context(hits) if hits else "(No relevant passages retrieved.)"

    # No API key → still show useful excerpts from the documents
    if not has_openai_key():
        analyst, auditor = _extractive_demo_answers(resolved, domain, hits)
        return {**state, "analyst": analyst, "auditor": auditor}

    try:
        llm = _llm()
        analyst_raw = llm.invoke(
            _persona_prompt("Analyst", domain, question, context, history, resolved)
        ).content
        auditor_raw = llm.invoke(
            _persona_prompt("Auditor", domain, question, context, history, resolved)
        ).content
        analyst = _parse_persona_json(analyst_raw, hits)
        auditor = _parse_persona_json(auditor_raw, hits)
        return {**state, "analyst": analyst, "auditor": auditor}
    except Exception as exc:
        return {**state, "error": f"Answer generation failed: {exc}", "done": True}


# ===================================================================
# 8. Self-correct (quality gate + branching signal)
# ===================================================================

def self_correct(state: GraphState) -> GraphState:
    """
    Check answer quality. If weak, set needs_correction=True so the graph
    routes back to generate_subqueries (self-correction branch).
    """
    logger.info("[%s] self_correct", state.get("job_id"))
    loop = int(state.get("correction_loop") or 0)
    analyst = state.get("analyst") or {}
    hits = state.get("ranked_hits") or []
    answer = (analyst.get("answer") or "").strip().lower()

    # Demo mode has fixed low confidence — retrying won't help without an LLM
    if not has_openai_key():
        return {
            **state,
            "needs_correction": False,
            "correction_reason": "",
            "done": True,
        }

    reasons: list[str] = []
    if not hits:
        reasons.append("No passages survived re-ranking — broaden search terms.")
    if float(analyst.get("confidence") or 0) < 0.5:
        reasons.append(f"Analyst confidence too low ({analyst.get('confidence')}).")
    if hits and not analyst.get("citations"):
        reasons.append("Hits exist but Analyst returned no citations — ground the answer.")
    if not answer:
        reasons.append("Analyst answer is empty.")

    # Evasive / non-answers even when we retrieved evidence
    evasive_markers = (
        "unclear",
        "lack of context",
        "lacks sufficient context",
        "no relevant context",
        "please provide",
        "cannot be generated",
        "not possible to",
        "ambiguous",
    )
    if hits and any(m in answer for m in evasive_markers):
        reasons.append(
            "Answer looks evasive despite retrieved passages — search with more "
            "specific document keywords and quote the passages."
        )

    # Very short generic answers with evidence available
    if hits and len(answer) < 80:
        reasons.append("Answer is too short given available passages.")

    needs = bool(reasons) and loop < settings.max_correction_loops

    return {
        **state,
        "needs_correction": needs,
        "correction_reason": " ".join(reasons),
        "correction_loop": loop + 1 if needs else loop,
        "done": not needs,
    }
