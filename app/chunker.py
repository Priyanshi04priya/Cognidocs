"""
Text chunker with paragraph/sentence-aware splits.

Target window: ~1000 characters with ~200 overlap (from settings).
Prefer breaking on paragraph or sentence boundaries so clauses stay intact.
"""

from __future__ import annotations

import re
from typing import Any

from app.config import settings


def _split_units(text: str) -> list[str]:
    """Split into paragraphs, then sentences — keeps natural document units."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        # If paragraph already short, keep whole
        if len(para) <= settings.chunk_size:
            units.append(para)
            continue
        # Split long paragraphs into sentences
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if not buf:
                buf = sent
            elif len(buf) + 1 + len(sent) <= settings.chunk_size:
                buf = f"{buf} {sent}"
            else:
                units.append(buf)
                buf = sent
        if buf:
            units.append(buf)
    return units or ([text.strip()] if text.strip() else [])


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    units = _split_units(text)
    chunks: list[str] = []
    current = ""

    for unit in units:
        # Oversized single unit → hard character window with overlap
        if len(unit) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(unit):
                piece = unit[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                if start + chunk_size >= len(unit):
                    break
                start += step
            continue

        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= chunk_size:
            current = f"{current}\n\n{unit}"
        else:
            chunks.append(current.strip())
            # Overlap: keep tail of previous chunk
            if overlap > 0 and len(current) > overlap:
                tail = current[-overlap:].strip()
                current = f"{tail}\n\n{unit}" if tail else unit
            else:
                current = unit

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Chunk a list of document sections.

    Input:  [{"text": "...", "source": "a.pdf", "page": 1}, ...]
    Output: [{"text": "...", "source": "a.pdf", "page": 1, "chunk_id": 0}, ...]
    """
    results: list[dict[str, Any]] = []
    chunk_id = 0

    for section in sections:
        pieces = chunk_text(section["text"])
        for piece in pieces:
            results.append(
                {
                    "text": piece,
                    "source": section.get("source", "unknown"),
                    "page": section.get("page", 0),
                    "chunk_id": chunk_id,
                }
            )
            chunk_id += 1

    return results
