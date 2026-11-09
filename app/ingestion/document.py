"""Format-independent document model produced by the PDF and DOCX extractors.

Everything downstream (parse-check, section splitter, structurer) works on
``Document``. Each ``Line`` knows its character offsets in ``Document.text``
so extracted fields can always point back to their source span.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points

FileType = Literal["pdf", "docx"]

# Typographic ligatures and invisible characters that break keyword matching
# ("Certiﬁed" != "Certified"). Private-use icon glyphs are kept on purpose so
# the parse-check can report them.
_CLEAN = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "­": None, "​": None, "‌": None, "‍": None, "⁠": None, "﻿": None,
    "	": "	",
})


def clean_text(text: str) -> str:
    return text.translate(_CLEAN)


@dataclass
class Line:
    text: str
    page: int = 0  # 0-based
    block: int = 0  # running block id; a blank line separates blocks in Document.text
    bbox: BBox | None = None
    font_size: float | None = None
    bold: bool = False
    fonts: tuple[str, ...] = ()
    column: int = 0  # 0 = single column / leftmost column
    in_table: bool = False
    in_header_footer: bool = False
    in_textbox: bool = False
    style: str | None = None  # DOCX paragraph style name, e.g. "Heading 1"
    list_item: bool = False  # DOCX numbered/bulleted paragraph (marker not in text)
    ocr: bool = False
    # Filled by Document.finalize():
    start: int = 0
    end: int = 0

    @property
    def stripped(self) -> str:
        return self.text.strip()


@dataclass
class ImageInfo:
    page: int
    bbox: BBox | None = None
    width: float = 0.0
    height: float = 0.0

    @property
    def is_icon_sized(self) -> bool:
        # Pictograms (phone, mail, pin...) are typically ≤ 24 pt square.
        return 0 < self.width <= 24 and 0 < self.height <= 24


@dataclass
class TableInfo:
    page: int
    rows: int
    cols: int
    bbox: BBox | None = None
    text: str = ""


@dataclass
class FontInfo:
    name: str
    kind: str = ""  # e.g. "TrueType", "Type1", "Type3"
    embedded: bool | None = None
    pages: set[int] = field(default_factory=set)


@dataclass
class PageInfo:
    number: int
    width: float = 0.0
    height: float = 0.0
    char_count: int = 0
    image_count: int = 0
    image_area_ratio: float = 0.0  # share of page area covered by images
    columns: int = 1
    ocr_used: bool = False


@dataclass
class Document:
    source: str
    file_type: FileType
    extractor: str
    lines: list[Line] = field(default_factory=list)
    pages: list[PageInfo] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)
    tables: list[TableInfo] = field(default_factory=list)
    fonts: list[FontInfo] = field(default_factory=list)
    # Text found in DOCX headers/footers (PDF repeated bands are flagged on lines).
    header_footer_text: list[str] = field(default_factory=list)
    # DOCX glyphs from symbol fonts (Wingdings, Font Awesome...) via <w:sym>.
    symbol_glyphs: int = 0
    # Page count stored by the authoring app (DOCX has no fixed pages).
    reported_pages: int | None = None
    # Things the extractor noticed but could not fix (e.g. "OCR unavailable").
    notes: list[str] = field(default_factory=list)
    text: str = ""

    def finalize(self) -> Document:
        """Clean text, drop empty lines and compute ``text`` plus per-line offsets."""
        for ln in self.lines:
            ln.text = clean_text(ln.text)
        self.lines = [ln for ln in self.lines if ln.text.strip()]
        parts: list[str] = []
        pos = 0
        prev_key: tuple[int, int] | None = None
        for ln in self.lines:
            key = (ln.page, ln.block)
            if parts:
                sep = "\n\n" if key != prev_key else "\n"
                parts.append(sep)
                pos += len(sep)
            ln.text = ln.text.rstrip()
            ln.start = pos
            ln.end = pos + len(ln.text)
            parts.append(ln.text)
            pos = ln.end
            prev_key = key
        self.text = "".join(parts)
        return self

    @property
    def page_count(self) -> int:
        return self.reported_pages or len(self.pages)

    def lines_on_page(self, page: int) -> list[Line]:
        return [ln for ln in self.lines if ln.page == page]
