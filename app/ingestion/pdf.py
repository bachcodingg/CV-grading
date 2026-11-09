"""PDF text extraction.

Primary: PyMuPDF with column-aware block ordering (see ``layout.py``).
Fallback: pdfplumber, used when PyMuPDF fails or finds no text.
Scanned pages: optional OCR with Tesseract (``pytesseract``) when installed.
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf

from app.config import IngestionConfig
from app.ingestion.document import Document, FontInfo, ImageInfo, Line, PageInfo, TableInfo
from app.ingestion.layout import LayoutBlock, column_count, order_blocks

log = logging.getLogger(__name__)

_BOLD_FLAG = 1 << 4
_BOLD_NAME = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)


def _is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & _BOLD_FLAG) or bool(_BOLD_NAME.search(span.get("font", "")))


def _inside(bbox, area, tol: float = 2.0) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return area[0] - tol <= cx <= area[2] + tol and area[1] - tol <= cy <= area[3] + tol


def _open(source: str | Path | bytes) -> pymupdf.Document:
    if isinstance(source, (bytes, bytearray)):
        return pymupdf.open(stream=bytes(source), filetype="pdf")
    return pymupdf.open(str(source))


# ---------------------------------------------------------------- PyMuPDF


def _page_lines_pymupdf(page: pymupdf.Page, page_no: int, block_base: int, cfg: IngestionConfig):
    """Return (lines, n_blocks, n_columns) for one page in reading order."""
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    raw_blocks = [b for b in data["blocks"] if b.get("type", 0) == 0]

    layout_blocks: list[LayoutBlock] = []
    by_id: dict[int, dict] = {}
    for b in raw_blocks:
        text = "\n".join("".join(s["text"] for s in ln["spans"]) for ln in b["lines"])
        if not text.strip():
            continue
        lb = LayoutBlock(bbox=tuple(b["bbox"]), text=text)
        layout_blocks.append(lb)
        by_id[id(lb)] = b

    ordered = order_blocks(layout_blocks, min_gutter=cfg.min_gutter_width)
    lines: list[Line] = []
    for offset, (lb, col) in enumerate(ordered):
        raw = by_id[id(lb)]
        for ln in sorted(raw["lines"], key=lambda l: (round(l["bbox"][1], 1), l["bbox"][0])):
            spans = [s for s in ln["spans"] if s["text"]]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans)
            bold_chars = sum(len(s["text"].strip()) for s in spans if _is_bold(s))
            all_chars = sum(len(s["text"].strip()) for s in spans) or 1
            lines.append(
                Line(
                    text=text,
                    page=page_no,
                    block=block_base + offset,
                    bbox=tuple(ln["bbox"]),
                    font_size=round(max(s["size"] for s in spans), 1),
                    bold=bold_chars / all_chars >= 0.5,
                    fonts=tuple(dict.fromkeys(s["font"] for s in spans)),
                    column=col,
                )
            )
    return lines, len(ordered), column_count(ordered)


def _ruling_segments(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is not None:
            rects.append((r.x0, r.y0, r.x1, r.y1))
    return rects


def _mark_tables(page: pymupdf.Page, page_no: int, lines: list[Line], doc: Document) -> None:
    """Flag ruled tables.

    Newer PyMuPDF versions also guess *borderless* tables from text
    alignment, which flags every CV with right-aligned dates. Only tables
    backed by vector ruling lines are kept; borderless layouts are handled
    by the column logic instead.
    """
    rulings = _ruling_segments(page)
    if not rulings:
        return
    try:
        found = page.find_tables(strategy="lines")
    except Exception as exc:  # table detection is best-effort
        log.debug("find_tables failed on page %s: %s", page_no, exc)
        return
    for tab in found.tables:
        if tab.row_count < 2 or tab.col_count < 2:
            continue
        if sum(1 for r in rulings if _inside(r, tab.bbox)) < 3:
            continue
        inside = [ln for ln in lines if ln.bbox and _inside(ln.bbox, tab.bbox)]
        if not inside:
            continue
        for ln in inside:
            ln.in_table = True
        doc.tables.append(
            TableInfo(
                page=page_no,
                rows=tab.row_count,
                cols=tab.col_count,
                bbox=tuple(tab.bbox),
                text="\n".join(ln.text for ln in inside),
            )
        )


def _mark_repeated_bands(doc: Document, cfg: IngestionConfig) -> None:
    """Flag text repeated in the top/bottom band of several pages (running header/footer)."""
    if len(doc.pages) < 2:
        return

    def key(text: str) -> str:
        return re.sub(r"\d+", "#", text.strip().lower())

    def in_band(ln: Line) -> bool:
        if not ln.bbox:
            return False
        h = doc.pages[ln.page].height or 1
        return ln.bbox[3] <= cfg.margin_band * h or ln.bbox[1] >= (1 - cfg.margin_band) * h

    counts = Counter()
    for ln in doc.lines:
        if in_band(ln):
            counts[key(ln.text)] += 1
    for ln in doc.lines:
        if in_band(ln) and counts[key(ln.text)] >= 2:
            ln.in_header_footer = True


def _collect_fonts(pdf: pymupdf.Document) -> list[FontInfo]:
    fonts: dict[str, FontInfo] = {}
    for pno in range(pdf.page_count):
        for xref, ext, ftype, basefont, *_ in pdf.get_page_fonts(pno):
            name = basefont.split("+", 1)[-1] or f"xref{xref}"
            info = fonts.setdefault(name, FontInfo(name=name, kind=ftype, embedded=ext != "n/a"))
            info.pages.add(pno)
    return list(fonts.values())


def _ocr_page(page: pymupdf.Page, page_no: int, block_base: int, cfg: IngestionConfig) -> list[Line] | None:
    """OCR one page with Tesseract. Returns None if OCR is not available."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        pix = page.get_pixmap(dpi=cfg.ocr_dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="eng")
    except Exception as exc:  # binary missing, language pack missing, ...
        log.info("OCR failed on page %s: %s", page_no, exc)
        return None
    lines: list[Line] = []
    block = block_base
    for para in re.split(r"\n\s*\n", text):
        for raw in para.splitlines():
            if raw.strip():
                lines.append(Line(text=raw, page=page_no, block=block, ocr=True))
        block += 1
    return lines


def extract_pdf_pymupdf(source: str | Path | bytes, cfg: IngestionConfig, name: str = "") -> Document:
    pdf = _open(source)
    doc = Document(source=name or str(source if not isinstance(source, bytes) else "<bytes>"),
                   file_type="pdf", extractor="pymupdf")
    try:
        block_base = 0
        for pno, page in enumerate(pdf):
            lines, nblocks, ncols = _page_lines_pymupdf(page, pno, block_base, cfg)
            block_base += nblocks + 1
            info = PageInfo(number=pno, width=page.rect.width, height=page.rect.height, columns=ncols)

            page_area = max(page.rect.width * page.rect.height, 1.0)
            covered = 0.0
            for img in page.get_image_info():
                bb = img.get("bbox")
                w, h = (bb[2] - bb[0], bb[3] - bb[1]) if bb else (0.0, 0.0)
                doc.images.append(ImageInfo(page=pno, bbox=tuple(bb) if bb else None, width=w, height=h))
                covered += max(w, 0) * max(h, 0)
                info.image_count += 1
            info.image_area_ratio = min(covered / page_area, 1.0)

            info.char_count = sum(len(ln.text.strip()) for ln in lines)
            if info.char_count < cfg.min_chars_per_page:
                ocr_lines = _ocr_page(page, pno, block_base, cfg)
                if ocr_lines is None:
                    if info.image_count:
                        doc.notes.append(f"page {pno + 1}: looks scanned but OCR (Tesseract) is not available")
                elif ocr_lines:
                    lines = ocr_lines
                    info.ocr_used = True
                    info.char_count = sum(len(ln.text.strip()) for ln in lines)
                    block_base += len({ln.block for ln in lines}) + 1

            if not info.ocr_used:
                _mark_tables(page, pno, lines, doc)
            doc.lines.extend(lines)
            doc.pages.append(info)
        doc.fonts = _collect_fonts(pdf)
    finally:
        pdf.close()
    _mark_repeated_bands(doc, cfg)
    if any(p.ocr_used for p in doc.pages):
        doc.extractor = "pymupdf+ocr"
    return doc.finalize()


# --------------------------------------------------------------- pdfplumber


def _plumber_segments(page, cfg: IngestionConfig) -> list[dict]:
    """Group words into line segments, splitting a visual line at wide gaps.

    pdfplumber's own line grouping joins words on the same baseline across
    columns ("CONTACT EXPERIENCE"); splitting at gaps keeps columns apart.
    """
    words = page.extract_words(extra_attrs=["fontname", "size"], keep_blank_chars=False)
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if rows and abs(rows[-1][0]["top"] - w["top"]) <= 3:
            rows[-1].append(w)
        else:
            rows.append([w])
    segments: list[dict] = []
    for row in rows:
        row.sort(key=lambda w: w["x0"])
        current: list[dict] = [row[0]]
        for prev, w in zip(row, row[1:]):
            gap = w["x0"] - prev["x1"]
            if gap > max(cfg.min_gutter_width, 1.5 * float(prev["size"])):
                segments.append(_segment(current))
                current = []
            current.append(w)
        segments.append(_segment(current))
    return segments


def _segment(words: list[dict]) -> dict:
    n_bold = sum(1 for w in words if _BOLD_NAME.search(w.get("fontname", "")))
    return {
        "text": " ".join(w["text"] for w in words),
        "bbox": (min(w["x0"] for w in words), min(w["top"] for w in words),
                 max(w["x1"] for w in words), max(w["bottom"] for w in words)),
        "size": round(max(float(w["size"]) for w in words), 1),
        "bold": n_bold >= 0.5 * len(words),
        "fonts": tuple(dict.fromkeys(w.get("fontname", "") for w in words)),
    }


def extract_pdf_pdfplumber(source: str | Path | bytes, cfg: IngestionConfig, name: str = "") -> Document:
    import pdfplumber

    stream = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else str(source)
    doc = Document(source=name or str(source if not isinstance(source, bytes) else "<bytes>"),
                   file_type="pdf", extractor="pdfplumber")
    with pdfplumber.open(stream) as pdf:
        block_base = 0
        for pno, page in enumerate(pdf.pages):
            raw_lines = _plumber_segments(page, cfg)
            blocks = [LayoutBlock(bbox=seg["bbox"], text=seg["text"]) for seg in raw_lines]
            by_id = {id(b): seg for b, seg in zip(blocks, raw_lines)}
            ordered = order_blocks(blocks, min_gutter=cfg.min_gutter_width)
            lines: list[Line] = []
            prev_col, prev_bottom, block = None, None, block_base
            for lb, col in ordered:
                seg = by_id[id(lb)]
                top, bottom = lb.y0, lb.y1
                # New block on column change or a vertical gap bigger than ~1 line.
                if prev_col is not None and (col != prev_col or top - prev_bottom > 0.8 * (bottom - top)):
                    block += 1
                prev_col, prev_bottom = col, bottom
                lines.append(
                    Line(
                        text=seg["text"],
                        page=pno,
                        block=block,
                        bbox=lb.bbox,
                        font_size=seg["size"],
                        bold=seg["bold"],
                        fonts=seg["fonts"],
                        column=col,
                    )
                )
            for tab in page.find_tables():
                rows = tab.extract()
                if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
                    continue
                inside = [ln for ln in lines if ln.bbox and _inside(ln.bbox, tab.bbox)]
                for ln in inside:
                    ln.in_table = True
                doc.tables.append(TableInfo(page=pno, rows=len(rows), cols=max(len(r) for r in rows),
                                            bbox=tuple(tab.bbox), text="\n".join(ln.text for ln in inside)))
            for img in page.images:
                w, h = float(img["x1"] - img["x0"]), float(img["bottom"] - img["top"])
                doc.images.append(ImageInfo(page=pno, bbox=(img["x0"], img["top"], img["x1"], img["bottom"]), width=w, height=h))
            area = max(float(page.width * page.height), 1.0)
            pimgs = [i for i in doc.images if i.page == pno]
            doc.pages.append(
                PageInfo(
                    number=pno,
                    width=float(page.width),
                    height=float(page.height),
                    char_count=sum(len(ln.text.strip()) for ln in lines),
                    image_count=len(pimgs),
                    image_area_ratio=min(sum(i.width * i.height for i in pimgs) / area, 1.0),
                    columns=column_count(ordered),
                )
            )
            doc.lines.extend(lines)
            block_base = block + 2
    _mark_repeated_bands(doc, cfg)
    return doc.finalize()


# ------------------------------------------------------------------- facade


def extract_pdf(source: str | Path | bytes, cfg: IngestionConfig | None = None, name: str = "") -> Document:
    """Extract a PDF with PyMuPDF, falling back to pdfplumber if needed."""
    cfg = cfg or IngestionConfig()
    primary_error: Exception | None = None
    try:
        doc = extract_pdf_pymupdf(source, cfg, name)
        if doc.text.strip():
            return doc
    except Exception as exc:
        primary_error = exc
        log.warning("PyMuPDF failed (%s); falling back to pdfplumber", exc)
        doc = None

    try:
        fallback = extract_pdf_pdfplumber(source, cfg, name)
    except Exception as exc:
        if doc is not None:
            return doc
        raise ValueError(f"could not read PDF: {primary_error or exc}") from exc
    if fallback.text.strip() or doc is None:
        fallback.notes.append(
            "PyMuPDF " + (f"failed: {primary_error}" if primary_error else "found no text") + "; used pdfplumber"
        )
        if doc is not None:
            fallback.notes.extend(doc.notes)
        return fallback
    return doc
