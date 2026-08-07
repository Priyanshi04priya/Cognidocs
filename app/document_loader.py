"""
Document loaders for PDF, DOCX, and EML files.

Each loader returns a list of plain-text pages/sections with source metadata.
"""

from __future__ import annotations

import email
import email.policy
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from docx import Document as DocxDocument


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".eml", ".doc"}


def load_document(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load a single file and return text chunks with metadata.

    Returns a list of dicts:
        {"text": "...", "source": "file.pdf", "page": 1}
    """
    # Resolve so Windows relative paths / mixed slashes still work
    path = Path(str(file_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(path)
    if ext in {".docx", ".doc"}:
        return _load_docx(path)
    if ext == ".eml":
        return _load_eml(path)

    raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or EML.")


def load_documents(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load multiple documents and concatenate their sections."""
    all_sections: list[dict[str, Any]] = []
    for fp in file_paths:
        all_sections.extend(load_document(fp))
    return all_sections


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append({"text": text, "source": path.name, "page": i})
    return sections


def _load_docx(path: Path) -> list[dict[str, Any]]:
    doc = DocxDocument(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return []
    # Treat the whole doc as one section (chunker will split later)
    return [{"text": "\n".join(paragraphs), "source": path.name, "page": 1}]


def _load_eml(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)

    parts = [
        f"From: {msg.get('From', '')}",
        f"To: {msg.get('To', '')}",
        f"Subject: {msg.get('Subject', '')}",
        f"Date: {msg.get('Date', '')}",
        "",
    ]

    body = _extract_email_body(msg)
    parts.append(body)

    text = "\n".join(parts).strip()
    if not text:
        return []
    return [{"text": text, "source": path.name, "page": 1}]


def _extract_email_body(msg: email.message.Message) -> str:
    """Pull the plain-text (or HTML-stripped) body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                continue
            if content_type == "text/plain":
                try:
                    return part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode("utf-8", errors="replace")
        return ""

    try:
        return msg.get_content()
    except Exception:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="replace")
        return str(msg.get_payload() or "")
