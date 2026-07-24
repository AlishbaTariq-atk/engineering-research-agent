from __future__ import annotations

import re

import fitz  # PyMuPDF
from bs4 import BeautifulSoup


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Best-effort full-text extraction. Returns "" on a corrupt/encrypted
    PDF rather than raising - a single bad PDF should fail that one
    document, not the whole ingestion run."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""


def clean_html(html: str) -> str:
    """Strip tags/scripts/styles down to readable text - used for RSS
    entries and doc pages that embed full HTML rather than plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def normalize_for_hash(text: str) -> str:
    """Collapse whitespace before hashing so trivial reformatting (e.g. a
    source re-serving the same abstract with different line wrapping)
    doesn't register as a content change and trigger a spurious version bump."""
    return re.sub(r"\s+", " ", text or "").strip()
