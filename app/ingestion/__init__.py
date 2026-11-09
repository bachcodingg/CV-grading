"""[1] Ingestion: turn a CV file into a ``Document`` (text + layout facts)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from app.config import IngestionConfig
from app.ingestion.document import Document, Line

__all__ = ["Document", "Line", "detect_type", "load_document"]


def detect_type(data: bytes, filename: str = "") -> str:
    """Return "pdf" or "docx" from the file's magic bytes (extension as a tie-break)."""
    if data[:5] == b"%PDF-" or b"%PDF-" in data[:1024]:
        return "pdf"
    if data[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ValueError("legacy .doc files are not supported; save the CV as .docx or PDF")
    suffix = Path(filename).suffix.lower()
    if suffix in {".pdf", ".docx"}:
        return suffix[1:]
    raise ValueError(f"unsupported file type for {filename or 'input'}: expected PDF or DOCX")


def load_document(
    source: str | Path | bytes, filename: str = "", cfg: IngestionConfig | None = None
) -> Document:
    """Extract a PDF or DOCX CV from a path or raw bytes (e.g. an upload)."""
    from app.ingestion.docx import extract_docx
    from app.ingestion.pdf import extract_pdf

    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        filename = filename or path.name
    else:
        data = bytes(source)
    kind = detect_type(data, filename)
    if kind == "pdf":
        return extract_pdf(data, cfg, name=filename)
    return extract_docx(data, cfg, name=filename)
