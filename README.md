# AI Document Query Engine

Beginner-friendly RAG system that answers questions over **PDF**, **DOCX**, and **EML** files.

- **FastAPI** — 2 REST endpoints (submit job + poll result)
- **Streamlit** — simple upload / ask UI
- **Qdrant** — per-job vector collections (tenant isolation)
- **LangGraph** — 8-node stateful workflow with self-correction
- **Celery + Redis** — background job queue
- **CrossEncoder** (`ms-marco-MiniLM-L-6-v2`) — re-ranks retrieved chunks
- **GPT-4.1 mini** — dual-persona answers (Analyst + Auditor)

## Architecture (8 LangGraph nodes)

```
1. ingest_documents
2. detect_domain          → Legal | Insurance | HR | Finance
3. chunk_and_index        → 1000 chars / 200 overlap → Qdrant job_{id}
4. generate_subqueries    → 3–5 sub-queries
5. retrieve               → top-5 semantic search per query (asyncio)
6. rerank_hits            → CrossEncoder + dedupe
7. generate_answers       → Analyst + Auditor JSON
8. self_correct           → if weak → loop back to step 4
```

## Quick start

### 1. Start Qdrant + Redis

```bash
docker compose up -d
```

### 2. Install Python deps

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure env

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

> Without an API key the pipeline still runs in **demo mode** (keyword domain
> detection + structured placeholder answers) so you can test retrieval locally.

### 4. Start the processes

**Easiest local mode (recommended, especially on Windows)** — Celery not required when `SYNC_JOBS=true`:

```bash
# Terminal 1 — API (also runs jobs in a background thread)
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run streamlit_app.py
```

Open http://localhost:8501

**UI flow**
1. Upload documents once (indexed into Qdrant)
2. Ask as many questions as you want on the **same** documents
3. Use sidebar **Clear session / new documents** to upload different files

**Optional Celery mode** — set `SYNC_JOBS=false` in `.env`, then also run:

```bash
# Windows:
celery -A app.celery_app.celery_app worker --loglevel=info --pool=solo

# Mac/Linux:
celery -A app.celery_app.celery_app worker --loglevel=info
```

Open http://localhost:8501 and upload a document.

## API

### `POST /jobs`

Multipart form:

| Field      | Type   | Description              |
|------------|--------|--------------------------|
| `question` | string | Your question            |
| `files`    | files  | PDF / DOCX / EML uploads |

Returns:

```json
{ "job_id": "...", "status": "queued", "message": "..." }
```

### `GET /jobs/{job_id}`

Returns status + dual-persona result when ready:

```json
{
  "job_id": "...",
  "status": "completed",
  "result": {
    "domain": "Legal",
    "analyst": {
      "answer": "...",
      "confidence": 0.82,
      "citations": [{"source": "contract.pdf", "clause": "...", "score": 0.71}],
      "justification": ["Step 1...", "Step 2..."]
    },
    "auditor": { "...": "..." }
  }
}
```

## Project layout

```
app/
  main.py              # FastAPI (2 endpoints)
  config.py            # Settings from .env
  models.py            # Pydantic schemas
  celery_app.py        # Celery + Redis
  tasks.py             # Background job
  document_loader.py   # PDF / DOCX / EML
  chunker.py           # 1000 / 200 chunking
  embeddings.py        # SentenceTransformer
  qdrant_store.py      # Per-job collections
  reranker.py          # CrossEncoder
  graph/
    state.py           # LangGraph state
    nodes.py           # 8 nodes
    workflow.py        # Graph + self-correction edges
streamlit_app.py       # Frontend
docker-compose.yml     # Qdrant + Redis
sample_docs/           # Example EML
```

## Notes for beginners

1. Each uploaded job gets its **own Qdrant collection** (`job_<uuid>`).
2. Self-correction re-runs sub-query generation up to `MAX_CORRECTION_LOOPS` (default 2).
3. Keep **Qdrant** running (`docker compose up -d`). Redis is optional when `SYNC_JOBS=true`.
4. First run downloads embedding + CrossEncoder models (one-time, can take a few minutes).
5. **Windows tip:** keep `SYNC_JOBS=true` in `.env` so you only need API + Streamlit + Qdrant.
6. If you use Celery on Windows:

```bash
celery -A app.celery_app.celery_app worker --loglevel=info --pool=solo
```
