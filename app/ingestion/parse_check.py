"""Parse-check: show the job seeker what an ATS parser gets out of their CV.

Each finding names a concrete, fixable cause of lost or scrambled text
(tables, text boxes, header contact info, icon fonts, image-only pages...).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from app.config import IngestionConfig
from app.ingestion.document import Document
from app.patterns import EMAIL, GARBLED, PRIVATE_USE, find_phones

Severity = Literal["error", "warning", "info"]
_ORDER = {"error": 0, "warning": 1, "info": 2}
_SYMBOL_FONT = re.compile(r"symbol|wingding|webding|dingbat|awesome|icon|material", re.IGNORECASE)


@dataclass
class Finding:
    code: str
    severity: Severity
    message: str
    advice: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class ParseReport:
    source: str
    file_type: str
    extractor: str
    pages: int
    characters: int
    words: int
    columns: int
    findings: list[Finding]
    parser_view: str  # the plain text a parser extracts, in reading order

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def to_dict(self) -> dict:
        return asdict(self)

    def format(self, show_text: bool = True) -> str:
        icon = {"error": "✗", "warning": "!", "info": "i"}
        out = [
            f"Parse-check: {self.source} ({self.file_type}, {self.pages} page(s), "
            f"{self.words} words, extractor={self.extractor})",
        ]
        if not self.findings:
            out.append("  ✓ No parsing problems found.")
        for f in self.findings:
            out.append(f"  [{icon[f.severity]}] {f.message}")
            if f.advice:
                out.append(f"      → {f.advice}")
            for ex in f.examples[:3]:
                out.append(f"      · {ex}")
        if show_text:
            out += ["", "─── What a parser extracts ───", self.parser_view or "(no text)"]
        return "\n".join(out)


def _short(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def check_document(doc: Document, cfg: IngestionConfig | None = None) -> ParseReport:
    cfg = cfg or IngestionConfig()
    findings: list[Finding] = []
    add = findings.append
    text = doc.text
    chars = len(re.sub(r"\s", "", text))

    # ---- no / little text -------------------------------------------------
    if chars < 100:
        add(Finding(
            "NO_TEXT", "error",
            "Almost no machine-readable text was found. The CV is probably a scan or exported as an image.",
            "Export the CV from Word / Google Docs / LaTeX as a text PDF, or upload the DOCX.",
        ))
    for page in doc.pages:
        if page.ocr_used:
            add(Finding(
                "OCR_USED", "warning",
                f"Page {page.number + 1} had no text layer and was read with OCR.",
                "Many ATS do not run OCR. Upload a PDF with selectable text.",
            ))
        elif chars >= 100 and page.char_count < cfg.min_chars_per_page and page.image_count:
            add(Finding(
                "IMAGE_ONLY_PAGE", "error",
                f"Page {page.number + 1} contains images but no readable text.",
                "Content on this page is invisible to parsers. Replace images of text with real text.",
            ))
        elif page.char_count >= cfg.min_chars_per_page and page.image_area_ratio > 0.25:
            add(Finding(
                "LARGE_IMAGE", "warning",
                f"About {page.image_area_ratio:.0%} of page {page.number + 1} is covered by images.",
                "Infographics, skill bars and logos carry no text for a parser. Make sure every "
                "important fact also appears as plain text.",
            ))

    # ---- hidden places ----------------------------------------------------
    table_lines = [ln for ln in doc.lines if ln.in_table]
    if table_lines:
        add(Finding(
            "TABLES", "warning",
            f"{len(doc.tables)} table(s) found; {len(table_lines)} line(s) of text sit inside table cells.",
            "Some ATS read tables cell by cell, mix up columns or skip them. Use plain paragraphs "
            "with tab-aligned dates instead.",
            [_short(ln.text) for ln in table_lines[:3]],
        ))
    box_lines = [ln for ln in doc.lines if ln.in_textbox]
    if box_lines:
        add(Finding(
            "TEXT_BOXES", "warning",
            f"{len(box_lines)} line(s) are inside text boxes or shapes.",
            "Many parsers ignore text boxes entirely. Move this text into the normal document body.",
            [_short(ln.text) for ln in box_lines[:3]],
        ))

    hf_lines = [ln for ln in doc.lines if ln.in_header_footer]
    hf_contact = [ln for ln in hf_lines if EMAIL.search(ln.text) or find_phones(ln.text)]
    if hf_contact:
        add(Finding(
            "CONTACT_IN_HEADER", "warning",
            "Contact details are in the page header/footer.",
            "Several ATS skip headers and footers, so your email or phone may be lost. "
            "Put contact details at the top of the body.",
            [_short(ln.text) for ln in hf_contact[:3]],
        ))
    elif any(not re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", ln.text.strip(), re.I) for ln in hf_lines):
        add(Finding(
            "HEADER_FOOTER_TEXT", "info",
            "Some text is in the page header/footer.",
            "Parsers may skip it; keep important information in the body.",
            [_short(ln.text) for ln in hf_lines[:3]],
        ))

    # ---- icons, symbols and garbled text ---------------------------------
    pua = PRIVATE_USE.findall(text)
    if pua or doc.symbol_glyphs:
        n = len(pua) + doc.symbol_glyphs
        add(Finding(
            "ICON_GLYPHS", "warning",
            f"{n} icon/symbol character(s) found (icon fonts such as Font Awesome or Wingdings).",
            "Parsers turn icons into junk characters or drop them. Use words (\"Email:\", \"Phone:\") "
            "or just the values.",
            [_short(ln.text) for ln in doc.lines if PRIVATE_USE.search(ln.text)][:3],
        ))
    icons = [img for img in doc.images if img.is_icon_sized]
    if len(icons) >= 2:
        add(Finding(
            "ICON_IMAGES", "info",
            f"{len(icons)} small pictogram images found (e.g. phone/mail icons).",
            "Icons are fine visually, but the information next to them must be plain text. "
            "Avoid icons that replace words such as skill names or levels.",
        ))
    garbled = GARBLED.findall(text)
    if garbled:
        ratio = len(garbled) / max(chars, 1)
        add(Finding(
            "GARBLED_TEXT", "error" if ratio > 0.02 else "warning",
            f"{len(garbled)} unreadable character(s) found; a font has no proper text mapping.",
            "Re-export the PDF with standard fonts (embed fonts, avoid 'print to PDF' from images), "
            "and check the parser view below for scrambled words.",
            [_short(ln.text) for ln in doc.lines if GARBLED.search(ln.text)][:3],
        ))

    type3 = [f.name for f in doc.fonts if f.kind == "Type3"]
    odd = sorted({f.name for f in doc.fonts if _SYMBOL_FONT.search(f.name)} | set(type3))
    if odd:
        add(Finding(
            "ODD_FONTS", "warning" if type3 else "info",
            "Unusual fonts found: " + ", ".join(odd[:5]) + (" …" if len(odd) > 5 else ""),
            "Type3 and symbol/icon fonts often extract as gibberish. Use common text fonts for content.",
        ))

    # ---- layout ----------------------------------------------------------
    multi = [p.number + 1 for p in doc.pages if p.columns > 1]
    if multi:
        add(Finding(
            "MULTI_COLUMN", "info",
            f"Multi-column layout detected on page(s) {', '.join(map(str, multi))}.",
            "We rebuilt the reading order column by column. Simpler ATS read straight across the page "
            "and can mix columns; check the parser view, or prefer a single-column layout.",
        ))

    # ---- content sanity ----------------------------------------------------
    if chars >= 100:
        if not EMAIL.search(text):
            add(Finding("NO_EMAIL", "warning", "No email address was found in the extracted text.",
                        "Recruiters and ATS need it. Write it out as plain text (not only as a link or icon)."))
        if not find_phones(text):
            add(Finding("NO_PHONE", "info", "No phone number was found in the extracted text."))
    if doc.page_count > cfg.max_pages_warning:
        add(Finding("LONG_CV", "info", f"The CV has {doc.page_count} pages.",
                    "Two pages is a common upper limit outside academia."))
    for note in doc.notes:
        add(Finding("EXTRACTOR_NOTE", "info", note))

    findings.sort(key=lambda f: _ORDER[f.severity])
    return ParseReport(
        source=doc.source,
        file_type=doc.file_type,
        extractor=doc.extractor,
        pages=doc.page_count,
        characters=chars,
        words=len(text.split()),
        columns=max((p.columns for p in doc.pages), default=1),
        findings=findings,
        parser_view=text,
    )
