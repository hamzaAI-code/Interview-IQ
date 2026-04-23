"""Extract plain text from a resume file (PDF / DOCX / TXT)."""
from __future__ import annotations

import io
from pathlib import Path


def parse_resume_bytes(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _from_pdf(data)
    if suffix in {".docx", ".doc"}:
        return _from_docx(data)
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported resume format: {suffix}")


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return _clean("\n".join(parts))


def _from_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return _clean("\n".join(p.text for p in doc.paragraphs))


def _clean(text: str) -> str:
    # collapse excessive whitespace; preserve paragraph breaks
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
