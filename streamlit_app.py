"""
Streamlit frontend — colourful UI with multi-question sessions.

Flow
----
1. Upload documents once  → creates a session (indexed in Qdrant)
2. Ask as many questions as you want on the SAME documents
3. Upload new docs anytime to start a fresh session

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import time

import requests
import streamlit as st

st.set_page_config(
    page_title="DocQuery AI",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colourful theme (teal + amber — not the usual purple AI look)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;700&family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
  --bg: #062a32;
  --bg2: #0b3d47;
  --panel: #0f4a56;
  --ink: #f3f7f8;
  --muted: #b7d0d5;
  --accent: #f0a202;
  --accent2: #2ec4b6;
  --analyst: #1b9aaa;
  --auditor: #e76f51;
  --ok: #2a9d8f;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, #1a6b78 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #7a4e12 0%, transparent 45%),
    linear-gradient(160deg, var(--bg) 0%, #041e24 100%);
  color: var(--ink);
  font-family: "Source Sans 3", sans-serif;
}

h1, h2, h3, .brand-title {
  font-family: "Fraunces", Georgia, serif !important;
  letter-spacing: -0.02em;
}

.hero {
  background: linear-gradient(135deg, rgba(46,196,182,0.18), rgba(240,162,2,0.14));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 22px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
}

.hero h1 {
  margin: 0 0 0.35rem 0;
  font-size: 2.2rem;
  color: #fff;
}

.hero p {
  margin: 0;
  color: #d7e8eb;
  font-size: 1.05rem;
}

.feature-banner {
  background: linear-gradient(90deg, #f0a202, #ffcc4d);
  color: #1a1200;
  border-radius: 16px;
  padding: 0.95rem 1.2rem;
  font-weight: 700;
  font-size: 1.05rem;
  margin: 0.8rem 0 1.2rem 0;
  box-shadow: 0 10px 30px rgba(240,162,2,0.25);
  animation: pulseGlow 2.8s ease-in-out infinite;
}

@keyframes pulseGlow {
  0%, 100% { transform: translateY(0); box-shadow: 0 10px 30px rgba(240,162,2,0.22); }
  50% { transform: translateY(-2px); box-shadow: 0 14px 36px rgba(240,162,2,0.35); }
}

.step-card {
  background: rgba(15, 74, 86, 0.72);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.8rem;
}

.step-num {
  display: inline-block;
  background: var(--accent2);
  color: #042226;
  font-weight: 700;
  border-radius: 999px;
  padding: 0.15rem 0.55rem;
  margin-right: 0.4rem;
}

.persona-analyst {
  background: linear-gradient(180deg, rgba(27,154,170,0.22), rgba(27,154,170,0.08));
  border: 1px solid rgba(27,154,170,0.45);
  border-radius: 18px;
  padding: 1rem;
}

.persona-auditor {
  background: linear-gradient(180deg, rgba(231,111,81,0.22), rgba(231,111,81,0.08));
  border: 1px solid rgba(231,111,81,0.45);
  border-radius: 18px;
  padding: 1rem;
}

.chip {
  display: inline-block;
  background: rgba(46,196,182,0.2);
  border: 1px solid rgba(46,196,182,0.45);
  color: #d9fffa;
  border-radius: 999px;
  padding: 0.2rem 0.7rem;
  margin: 0.15rem 0.25rem 0.15rem 0;
  font-size: 0.85rem;
}

.qa-bubble {
  background: rgba(255,255,255,0.06);
  border-left: 4px solid var(--accent);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  margin: 0.7rem 0;
}

div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #07343d, #041e24);
}

.stButton > button {
  border-radius: 12px !important;
  font-weight: 700 !important;
}

.stTextInput input, .stTextArea textarea {
  border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "doc_session_id" not in st.session_state:
    st.session_state.doc_session_id = None
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = {}
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

API_URL = st.sidebar.text_input("API URL", value=st.session_state.api_url)
st.session_state.api_url = API_URL


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def check_api() -> dict:
    resp = requests.get(f"{API_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def create_session(files) -> str:
    multipart = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in files
    ]
    resp = requests.post(f"{API_URL}/sessions", files=multipart, timeout=300)
    resp.raise_for_status()
    return resp.json()["session_id"]


def poll_session(session_id: str, timeout_s: int = 600) -> dict:
    start = time.time()
    box = st.empty()
    while time.time() - start < timeout_s:
        resp = requests.get(f"{API_URL}/sessions/{session_id}", timeout=60)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        box.info(f"Indexing documents… **{status}** ({int(time.time() - start)}s)")
        if status in {"ready", "failed"}:
            return data
        time.sleep(1.5)
    raise TimeoutError("Timed out while indexing documents.")


def ask_question(session_id: str, question: str, history: list[dict] | None = None) -> str:
    payload = {"question": question, "history": history or []}
    resp = requests.post(
        f"{API_URL}/sessions/{session_id}/ask",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["ask_id"]


def poll_ask(ask_id: str, timeout_s: int = 900) -> dict:
    start = time.time()
    box = st.empty()
    while time.time() - start < timeout_s:
        resp = requests.get(f"{API_URL}/asks/{ask_id}", timeout=60)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        box.info(f"Answering… **{status}** ({int(time.time() - start)}s)")
        if status in {"completed", "failed"}:
            return data
        time.sleep(1.5)
    raise TimeoutError("Timed out waiting for the answer.")


def render_persona(title: str, persona: dict | None, css_class: str) -> None:
    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
    if not persona:
        st.warning(f"No {title} answer.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    conf = float(persona.get("confidence") or 0)
    st.subheader(title)
    st.metric("Confidence", f"{conf:.0%}")
    st.markdown(persona.get("answer") or "—")

    just = persona.get("justification") or []
    if just:
        with st.expander("Step-by-step justification", expanded=False):
            for i, step in enumerate(just, 1):
                st.markdown(f"{i}. {step}")

    cites = persona.get("citations") or []
    if cites:
        with st.expander(f"Citations ({len(cites)})", expanded=False):
            for c in cites:
                st.markdown(
                    f"**{c.get('source', 'unknown')}** "
                    f"(score {float(c.get('score') or 0):.3f})"
                )
                st.code(c.get("clause") or "", language=None)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero + highlighted multi-ask feature
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>DocQuery AI</h1>
  <p>Upload PDF, DOCX or EML once — then keep asking questions on the same documents.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="feature-banner">
  ✨ Highlighted feature: Ask <u>multiple questions</u> from the <u>same document</u> —
  upload once, chat many times. No need to re-upload for every question!
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="step-card"><span class="step-num">1</span> Upload documents</div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="step-card"><span class="step-num">2</span> Wait until indexed</div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="step-card"><span class="step-num">3</span> Ask many questions</div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("### About")
    st.caption(
        "Domains auto-detected: Legal · Insurance · HR · Finance. "
        "Answers come from Analyst + Auditor personas."
    )
    st.markdown("### Sample files")
    st.caption("`sample_docs/hr_policy.eml` and `sample_service_agreement.docx`")
    if st.button("Clear session / new documents", use_container_width=True):
        st.session_state.doc_session_id = None
        st.session_state.doc_meta = {}
        st.session_state.qa_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# STEP 1 — Upload (only if no active session)
# ---------------------------------------------------------------------------
if not st.session_state.doc_session_id:
    st.markdown("### Step 1 — Upload your documents")
    uploaded = st.file_uploader(
        "PDF / DOCX / EML",
        type=["pdf", "docx", "eml", "doc"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", type="primary", use_container_width=True):
        if not uploaded:
            st.error("Please upload at least one document.")
        else:
            try:
                with st.spinner("Checking API..."):
                    check_api()
                with st.spinner("Uploading..."):
                    session_id = create_session(uploaded)
                meta = poll_session(session_id)
                if meta.get("status") == "failed":
                    st.error(meta.get("error") or "Indexing failed.")
                else:
                    st.session_state.doc_session_id = session_id
                    st.session_state.doc_meta = meta
                    st.session_state.qa_history = []
                    st.success("Documents indexed! You can now ask multiple questions.")
                    st.rerun()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot reach API at `{API_URL}`. "
                    "Start it with: `uvicorn app.main:app --reload --port 8000`"
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

else:
    # Active session banner
    meta = st.session_state.doc_meta or {}
    st.markdown("### Active document session")
    st.success(
        "Same documents are locked in — previous answers stay on this page. "
        "Follow-ups like “describe it” use earlier questions as context."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Domain", meta.get("domain") or "—")
    m2.metric("Chunks indexed", meta.get("num_indexed") or 0)
    m3.metric("Questions asked", len(st.session_state.qa_history))

    files = meta.get("files") or []
    if files:
        st.markdown(
            "".join(f'<span class="chip">📄 {name}</span>' for name in files),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="feature-banner" style="font-size:0.98rem;">'
        "🔁 Multi-ask + memory ON — ask follow-ups on the same docs; older Q&A stays visible below."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Always show previous Q&A first (so they never disappear) ----
    if st.session_state.qa_history:
        st.markdown("### Your Q&A on these documents")
        # Chronological: oldest → newest
        for i, item in enumerate(st.session_state.qa_history):
            result = item["result"]
            resolved = (result.get("metadata") or {}).get("resolved_question")
            st.markdown(
                f'<div class="qa-bubble"><strong>Q{i + 1}:</strong> {item["question"]}</div>',
                unsafe_allow_html=True,
            )
            if resolved and resolved.strip() != item["question"].strip():
                st.caption(f"Understood as: {resolved}")

            left, right = st.columns(2)
            with left:
                render_persona("Analyst", result.get("analyst"), "persona-analyst")
            with right:
                render_persona("Auditor", result.get("auditor"), "persona-auditor")

            meta_r = result.get("metadata") or {}
            with st.expander(f"Details for Q{i + 1}"):
                st.write(
                    {
                        "domain": result.get("domain"),
                        "resolved_question": meta_r.get("resolved_question"),
                        "ranked_hits": meta_r.get("num_ranked_hits"),
                        "sub_queries": meta_r.get("sub_queries"),
                    }
                )
            st.divider()

    # ---- Ask box at the bottom ----
    st.markdown("### Ask the next question")
    st.caption(
        "Tip: you can say “describe it in detail” or “explain more” — "
        "the app remembers your earlier questions in this session."
    )

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            placeholder="e.g. Explain Cognidocs in brief  /  describe it whole  /  What are the key features?",
            height=100,
        )
        asked = st.form_submit_button(
            "Ask this question",
            type="primary",
            use_container_width=True,
        )

    if asked:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            try:
                # Build chat history from prior answers for follow-up understanding
                history_payload = []
                for item in st.session_state.qa_history:
                    ans = ((item.get("result") or {}).get("analyst") or {}).get("answer") or ""
                    history_payload.append(
                        {"question": item["question"], "answer": ans[:800]}
                    )

                with st.spinner("Answering (previous Q&A stay on this page)..."):
                    ask_id = ask_question(
                        st.session_state.doc_session_id,
                        question.strip(),
                        history=history_payload,
                    )
                    payload = poll_ask(ask_id)

                result = payload.get("result") or {}
                if result.get("status") == "failed" or payload.get("status") == "failed":
                    st.error(result.get("error") or "Question failed.")
                else:
                    # Append so older answers remain above
                    st.session_state.qa_history.append(
                        {"question": question.strip(), "result": result}
                    )
                    st.rerun()
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at `{API_URL}`.")
            except requests.exceptions.ReadTimeout:
                st.error("API timed out. Check uvicorn / Qdrant and try again.")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
