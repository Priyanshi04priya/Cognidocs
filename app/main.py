"""
FastAPI entrypoint.

Endpoints
---------
  POST /sessions              → upload docs once (index into Qdrant)
  GET  /sessions/{id}         → session status (domain, files, ready?)
  POST /sessions/{id}/ask     → ask a question on the same docs
  GET  /asks/{ask_id}         → poll ask result

  POST /jobs                  → legacy one-shot upload+question
  GET  /jobs/{job_id}         → poll one-shot job

Default: SYNC_JOBS=true runs work in local background threads (easy on Windows).
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.document_loader import SUPPORTED_EXTENSIONS
from app.models import (
    AskRequest,
    AskSubmitResponse,
    JobStatusResponse,
    JobSubmitResponse,
    SessionCreateResponse,
    SessionStatusResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Document Query Engine",
    description=(
        "RAG pipeline over PDF / DOCX / EML with domain auto-detection, "
        "Qdrant retrieval, CrossEncoder re-ranking, and dual-persona answers. "
        "Upload once, ask many questions."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_local_jobs: dict[str, dict[str, Any]] = {}
_sessions: dict[str, dict[str, Any]] = {}
_asks: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _safe_filename(name: str) -> str:
    base = name.replace("\\", "/").split("/")[-1]
    stem = Path(base).stem
    ext = Path(base).suffix.lower()
    stem = re.sub(r"[^\w.\- ]+", "_", stem).strip() or "upload"
    return f"{stem}{ext}"


def _set_job(job_id: str, payload: dict[str, Any]) -> None:
    with _lock:
        _local_jobs[job_id] = payload


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        return _local_jobs.get(job_id)


def _set_session(session_id: str, payload: dict[str, Any]) -> None:
    with _lock:
        _sessions[session_id] = payload


def _get_session(session_id: str) -> dict[str, Any] | None:
    with _lock:
        return _sessions.get(session_id)


def _set_ask(ask_id: str, payload: dict[str, Any]) -> None:
    with _lock:
        _asks[ask_id] = payload


def _get_ask(ask_id: str) -> dict[str, Any] | None:
    with _lock:
        return _asks.get(ask_id)


def _save_uploads(files: list[UploadFile], folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload in files:
        name = _safe_filename(upload.filename or "upload.bin")
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
            )
        dest = (folder / name).resolve()
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        saved.append(str(dest))
    return saved


def _shape_ask_result(ask_id: str, question: str, final: dict[str, Any], files: list[str]) -> dict:
    if final.get("error"):
        return {
            "job_id": ask_id,
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
        "job_id": ask_id,
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
            "files": files,
        },
    }


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

def _run_local_job(job_id: str, question: str, file_paths: list[str]) -> None:
    from app.tasks import execute_query_job

    _set_job(job_id, {"job_id": job_id, "status": "running", "question": question})
    result = execute_query_job(job_id=job_id, question=question, file_paths=file_paths)
    _set_job(job_id, result)


def _run_session_index(session_id: str, file_paths: list[str], file_names: list[str]) -> None:
    from app.graph.workflow import run_ingest_only

    _set_session(
        session_id,
        {
            "session_id": session_id,
            "status": "indexing",
            "files": file_names,
            "file_paths": file_paths,
            "domain": None,
            "num_indexed": 0,
            "error": None,
        },
    )
    final = run_ingest_only(job_id=session_id, file_paths=file_paths)
    if final.get("error"):
        _set_session(
            session_id,
            {
                "session_id": session_id,
                "status": "failed",
                "files": file_names,
                "file_paths": file_paths,
                "domain": final.get("domain"),
                "num_indexed": 0,
                "error": final["error"],
            },
        )
        return

    _set_session(
        session_id,
        {
            "session_id": session_id,
            "status": "ready",
            "files": file_names,
            "file_paths": file_paths,
            "domain": final.get("domain"),
            "num_indexed": final.get("num_indexed", 0),
            "error": None,
        },
    )


def _run_ask_job(
    ask_id: str,
    session_id: str,
    question: str,
    domain: str,
    files: list[str],
    history: list[dict] | None = None,
) -> None:
    from app.graph.workflow import run_ask_only

    _set_ask(ask_id, {"job_id": ask_id, "status": "running", "question": question, "domain": domain})
    final = run_ask_only(
        job_id=session_id,
        question=question,
        domain=domain,
        chat_history=history or [],
    )
    shaped = _shape_ask_result(ask_id, question, final, files)
    # Surface how the follow-up was understood
    meta = dict(shaped.get("metadata") or {})
    meta["resolved_question"] = final.get("resolved_question") or question
    shaped["metadata"] = meta
    _set_ask(ask_id, shaped)


# ---------------------------------------------------------------------------
# Startup / health
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _ensure_upload_dir() -> None:
    settings.upload_path.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "upload_dir": str(settings.upload_path),
        "sync_jobs": settings.sync_jobs,
    }


# ---------------------------------------------------------------------------
# Sessions = upload once, ask many times
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(
    files: list[UploadFile] = File(..., description="PDF, DOCX, or EML files"),
) -> SessionCreateResponse:
    """Upload documents once. After status=ready, ask many questions."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    session_id = str(uuid.uuid4())
    folder = settings.upload_path / session_id
    saved_paths = _save_uploads(files, folder)
    file_names = [Path(p).name for p in saved_paths]

    _set_session(
        session_id,
        {
            "session_id": session_id,
            "status": "queued",
            "files": file_names,
            "file_paths": saved_paths,
            "domain": None,
            "num_indexed": 0,
            "error": None,
        },
    )
    threading.Thread(
        target=_run_session_index,
        args=(session_id, saved_paths, file_names),
        daemon=True,
        name=f"session-{session_id[:8]}",
    ).start()

    return SessionCreateResponse(session_id=session_id, status="indexing")


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
def get_session(session_id: str) -> SessionStatusResponse:
    session = _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionStatusResponse(
        session_id=session_id,
        status=session.get("status", "unknown"),
        domain=session.get("domain"),
        files=session.get("files") or [],
        num_indexed=int(session.get("num_indexed") or 0),
        error=session.get("error"),
    )


@app.post("/sessions/{session_id}/ask", response_model=AskSubmitResponse)
def ask_session(session_id: str, body: AskRequest) -> AskSubmitResponse:
    """Ask another question on the same already-indexed documents."""
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session = _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Upload documents first.")
    if session.get("status") != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Session is '{session.get('status')}'. Wait until status is 'ready'.",
        )

    ask_id = str(uuid.uuid4())
    history = [
        {"question": h.get("question", ""), "answer": h.get("answer", "")}
        for h in (body.history or [])
        if (h.get("question") or h.get("answer"))
    ]
    _set_ask(ask_id, {"job_id": ask_id, "status": "queued", "question": question})
    threading.Thread(
        target=_run_ask_job,
        args=(
            ask_id,
            session_id,
            question,
            session.get("domain") or "General",
            session.get("files") or [],
            history,
        ),
        daemon=True,
        name=f"ask-{ask_id[:8]}",
    ).start()

    return AskSubmitResponse(ask_id=ask_id, session_id=session_id, status="queued")


@app.get("/asks/{ask_id}", response_model=JobStatusResponse)
def get_ask(ask_id: str) -> JobStatusResponse:
    payload = _get_ask(ask_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Ask not found.")
    status = payload.get("status", "pending")
    if status in {"queued", "running", "pending"}:
        return JobStatusResponse(job_id=ask_id, status=status)
    return JobStatusResponse(job_id=ask_id, status=status, result=payload)


# ---------------------------------------------------------------------------
# Legacy one-shot /jobs (still supported)
# ---------------------------------------------------------------------------

@app.post("/jobs", response_model=JobSubmitResponse)
async def submit_job(
    question: str = Form(..., description="Your question about the documents"),
    files: list[UploadFile] = File(..., description="PDF, DOCX, or EML files"),
) -> JobSubmitResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    job_id = str(uuid.uuid4())
    saved_paths = _save_uploads(files, settings.upload_path / job_id)
    q = question.strip()

    if settings.sync_jobs:
        _set_job(job_id, {"job_id": job_id, "status": "queued", "question": q})
        threading.Thread(
            target=_run_local_job,
            args=(job_id, q, saved_paths),
            daemon=True,
            name=f"job-{job_id[:8]}",
        ).start()
        return JobSubmitResponse(
            job_id=job_id,
            status="queued",
            message="Job accepted (local background thread).",
        )

    try:
        from app.tasks import run_query_job

        run_query_job.apply_async(
            kwargs={"job_id": job_id, "question": q, "file_paths": saved_paths},
            task_id=job_id,
        )
    except Exception as exc:
        logger.warning("Celery enqueue failed (%s); falling back to local thread", exc)
        _set_job(job_id, {"job_id": job_id, "status": "queued", "question": q})
        threading.Thread(
            target=_run_local_job,
            args=(job_id, q, saved_paths),
            daemon=True,
            name=f"job-{job_id[:8]}",
        ).start()
        return JobSubmitResponse(
            job_id=job_id,
            status="queued",
            message="Celery unavailable — ran with local background thread instead.",
        )

    return JobSubmitResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    local = _get_job(job_id)
    if local is not None:
        status = local.get("status", "pending")
        if status in {"queued", "running", "pending"}:
            return JobStatusResponse(job_id=job_id, status=status)
        return JobStatusResponse(job_id=job_id, status=status, result=local)

    try:
        from celery.result import AsyncResult
        from app.tasks import run_query_job

        result = AsyncResult(job_id, app=run_query_job.app)
    except Exception as exc:
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            result={"job_id": job_id, "status": "failed", "error": str(exc)},
        )

    if result.state == "PENDING":
        return JobStatusResponse(job_id=job_id, status="pending")
    if result.state in {"STARTED", "RETRY"}:
        return JobStatusResponse(job_id=job_id, status="running")
    if result.state == "FAILURE":
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            result={"job_id": job_id, "status": "failed", "error": str(result.result)},
        )
    if result.state == "SUCCESS":
        payload = result.result or {}
        return JobStatusResponse(
            job_id=job_id,
            status=payload.get("status", "completed"),
            result=payload,
        )
    return JobStatusResponse(job_id=job_id, status=str(result.state).lower())
