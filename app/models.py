"""Pydantic request/response models for the API."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class JobSubmitResponse(BaseModel):
    """Returned immediately when a job is queued."""

    job_id: str
    status: str = "queued"
    message: str = "Job accepted and queued for processing."


class Citation(BaseModel):
    """A single clause-level citation from the source documents."""

    source: str = Field(description="File name the text came from")
    clause: str = Field(description="Exact text snippet used as evidence")
    score: float = Field(description="Relevance score after re-ranking")


class PersonaAnswer(BaseModel):
    """One persona's answer (Analyst or Auditor)."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    justification: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning",
    )


class JobResult(BaseModel):
    """Final structured JSON returned when a job / ask finishes."""

    job_id: str
    status: str
    domain: Optional[str] = None
    question: Optional[str] = None
    analyst: Optional[PersonaAnswer] = None
    auditor: Optional[PersonaAnswer] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    """Polling response for GET /jobs/{job_id} or GET /asks/{ask_id}."""

    job_id: str
    status: str
    result: Optional[JobResult] = None


class SessionCreateResponse(BaseModel):
    """Returned when documents are uploaded for multi-ask use."""

    session_id: str
    status: str = "indexing"
    message: str = "Documents accepted. Indexing in background."


class SessionStatusResponse(BaseModel):
    """Status of an uploaded document session."""

    session_id: str
    status: str
    domain: Optional[str] = None
    files: list[str] = Field(default_factory=list)
    num_indexed: int = 0
    error: Optional[str] = None


class AskRequest(BaseModel):
    """Ask a question against an already-indexed session."""

    question: str
    # Optional prior turns so follow-ups like "describe it" work
    history: list[dict[str, str]] = Field(default_factory=list)


class AskSubmitResponse(BaseModel):
    ask_id: str
    session_id: str
    status: str = "queued"
    message: str = "Question accepted."
