"""DOCX text extraction with python-docx.

The body is walked in document order so paragraphs and tables keep their
original sequence. Content that many ATS parsers miss is still extracted
(a human reader sees it) but flagged on the line, so the parse-check can
warn about it:

- table cells            -> ``Line.in_table``
- text boxes / shapes    -> ``Line.in_textbox``
- page headers / footers -> ``Line.in_header_footer``
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from docx import Document as open_docx
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from app.config import IngestionConfig
from app.ingestion.document import Document, FontInfo, ImageInfo, Line, PageInfo, TableInfo

_EMU_PER_PT = 12700
_W_P, _W_TBL = qn("w:p"), qn("w:tbl")
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "v": "urn:schemas-microsoft-com:vml",
}


def _style_attr(style, attr: str):
    """Look up a font attribute through the style inheritance chain."""
    while style is not None:
        value = getattr(style.font, attr)
        if value is not None:
            return value
        style = style.base_style
    return None


class _Builder:
    def __init__(self, doc: Document, docx) -> None:
        self.doc = doc
        self.docx = docx
        self.block = 0
        self.fonts: dict[str, FontInfo] = {}

    def new_block(self) -> None:
        self.block += 1

    # ---------------------------------------------------------- paragraphs

    def paragraph(self, p: Paragraph, **flags) -> None:
        text = p.text
        style = p.style.name if p.style is not None else None
        if not text.strip():
            self.new_block()
        else:
            runs = [r for r in p.runs if r.text.strip()]
            style_size = _style_attr(p.style, "size")
            sizes = [r.font.size.pt for r in runs if r.font.size is not None]
            if not sizes and style_size is not None:
                sizes = [style_size.pt]
            style_bold = bool(_style_attr(p.style, "bold"))
            bold_chars = sum(len(r.text.strip()) for r in runs if r.bold or (r.bold is None and style_bold))
            all_chars = sum(len(r.text.strip()) for r in runs) or 1
            names = tuple(dict.fromkeys(r.font.name for r in runs if r.font.name))
            for name in names:
                self.fonts.setdefault(name, FontInfo(name=name, kind="docx", embedded=None, pages={0}))
            is_heading = bool(style and re.match(r"(heading|title)", style, re.IGNORECASE))
            if is_heading:
                self.new_block()
            self.doc.lines.append(
                Line(
                    text=text,
                    block=self.block,
                    font_size=max(sizes) if sizes else None,
                    bold=bold_chars / all_chars >= 0.5 or (is_heading and not runs),
                    fonts=names,
                    style=style,
                    list_item=p._p.pPr is not None and p._p.pPr.numPr is not None
                    or bool(style and "list" in style.lower()),
                    **flags,
                )
            )
        self.textboxes(p._p, **flags)

    def textboxes(self, element, **flags) -> None:
        """Emit paragraphs inside text boxes anchored in ``element``."""
        boxes = [
            tb
            for tb in element.iter(qn("w:txbxContent"))
            if not any(anc.tag == _MC_FALLBACK for anc in tb.iterancestors())
        ]
        for tb in boxes:
            self.new_block()
            for p_el in tb.iterchildren(_W_P):
                self.paragraph(Paragraph(p_el, self.docx), **{**flags, "in_textbox": True})
            self.new_block()

    # --------------------------------------------------------------- tables

    def table(self, tbl, **flags) -> None:
        from docx.table import Table

        table = Table(tbl, self.docx)
        self.new_block()
        start = len(self.doc.lines)
        ncols = 0
        for row in table.rows:
            seen: set[int] = set()
            cells = []
            for cell in row.cells:  # merged cells repeat; keep each once
                if id(cell._tc) in seen:
                    continue
                seen.add(id(cell._tc))
                cells.append(cell)
            ncols = max(ncols, len(cells))
            for cell in cells:
                for child in cell._tc.iterchildren():
                    if child.tag == _W_P:
                        self.paragraph(Paragraph(child, cell), **{**flags, "in_table": True})
                    elif child.tag == _W_TBL:
                        self.table(child, **flags)
            self.new_block()
        lines = self.doc.lines[start:]
        self.doc.tables.append(
            TableInfo(page=0, rows=len(table.rows), cols=ncols, text="\n".join(ln.text for ln in lines))
        )

    def body(self, element, **flags) -> None:
        for child in element.iterchildren():
            if child.tag == _W_P:
                self.paragraph(Paragraph(child, self.docx), **flags)
            elif child.tag == _W_TBL:
                self.table(child, **flags)


def _images(root) -> list[ImageInfo]:
    images = []
    for ext in root.iterfind(".//wp:extent", _NS):
        w = int(ext.get("cx", 0)) / _EMU_PER_PT
        h = int(ext.get("cy", 0)) / _EMU_PER_PT
        holder = ext.getparent()
        if holder is not None and holder.find(".//a:blip", _NS) is not None:
            images.append(ImageInfo(page=0, width=w, height=h))
    for img in root.iterfind(".//v:imagedata", _NS):
        if not any(anc.tag == _MC_FALLBACK for anc in img.iterancestors()):
            images.append(ImageInfo(page=0))
    return images


def _reported_pages(docx) -> int | None:
    for rel in docx.part.package.iter_parts():
        if str(rel.partname) == "/docProps/app.xml":
            m = re.search(rb"<(?:\w+:)?Pages>(\d+)</", rel.blob)
            return int(m.group(1)) if m else None
    return None


def extract_docx(source: str | Path | bytes, cfg: IngestionConfig | None = None, name: str = "") -> Document:
    stream = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else str(source)
    try:
        docx = open_docx(stream)
    except Exception as exc:
        raise ValueError(f"could not read DOCX: {exc}") from exc

    doc = Document(source=name or str(source if not isinstance(source, bytes) else "<bytes>"),
                   file_type="docx", extractor="python-docx")
    b = _Builder(doc, docx)

    # Headers first, body, then footers: the reading order a human would use.
    seen_parts: set[int] = set()
    hf_elements = []
    headers, footers = [], []
    for section in docx.sections:
        for attr in ("first_page_header", "header", "even_page_header"):
            headers.append(getattr(section, attr))
        for attr in ("first_page_footer", "footer", "even_page_footer"):
            footers.append(getattr(section, attr))

    def emit_hf(parts) -> None:
        for hf in parts:
            if hf.is_linked_to_previous:
                continue
            el = hf._element
            if id(el) in seen_parts:
                continue
            seen_parts.add(id(el))
            hf_elements.append(el)
            before = len(doc.lines)
            b.new_block()
            b.body(el, in_header_footer=True)
            doc.header_footer_text.extend(ln.text for ln in doc.lines[before:])

    emit_hf(headers)
    b.new_block()
    body = docx.element.body
    b.body(body)
    b.new_block()
    emit_hf(footers)

    # Photos and contact icons often live in the header, so scan it too.
    roots = [body, *hf_elements]
    doc.images = [img for root in roots for img in _images(root)]
    doc.symbol_glyphs = sum(1 for root in roots for _ in root.iter(qn("w:sym")))
    doc.fonts = list(b.fonts.values())
    doc.reported_pages = _reported_pages(docx)
    doc.pages = [
        PageInfo(
            number=0,
            char_count=sum(len(ln.text.strip()) for ln in doc.lines),
            image_count=len(doc.images),
        )
    ]
    return doc.finalize()
