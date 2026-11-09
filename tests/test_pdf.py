import pytest

from app.config import IngestionConfig
from app.ingestion.pdf import extract_pdf, extract_pdf_pdfplumber, extract_pdf_pymupdf


@pytest.mark.parametrize("backend", [extract_pdf_pymupdf, extract_pdf_pdfplumber])
def test_single_column_reading_order(single_column_pdf, backend):
    doc = backend(single_column_pdf, IngestionConfig())
    text = doc.text
    assert text.index("Jane Doe") < text.index("EXPERIENCE") < text.index("EDUCATION")
    # A right-aligned date stays next to its job title.
    lines = [ln.text for ln in doc.lines]
    i = lines.index("Senior Software Engineer — Acme GmbH")
    assert lines[i + 1] == "Jan 2021 – Present"
    assert doc.pages[0].columns == 1
    assert not doc.tables


@pytest.mark.parametrize("backend", [extract_pdf_pymupdf, extract_pdf_pdfplumber])
def test_two_column_sidebar_is_not_interleaved(two_column_pdf, backend):
    doc = backend(two_column_pdf, IngestionConfig())
    text = doc.text
    # Entire sidebar first, then the main column.
    assert text.index("German (native)") < text.index("EXPERIENCE")
    assert text.index("Senior Software Engineer") < text.index("Acme GmbH | Jan 2021")
    assert doc.pages[0].columns == 2


def test_bold_and_font_size_are_captured(single_column_pdf):
    doc = extract_pdf(single_column_pdf)
    heading = next(ln for ln in doc.lines if ln.text == "EXPERIENCE")
    body = next(ln for ln in doc.lines if ln.text.startswith("Backend engineer"))
    assert heading.bold and not body.bold
    name = doc.lines[0]
    assert name.text == "Jane Doe" and name.font_size > heading.font_size


def test_ruled_table_is_flagged(table_pdf):
    doc = extract_pdf(table_pdf)
    assert len(doc.tables) == 1
    assert doc.tables[0].rows == 3
    flagged = {ln.text for ln in doc.lines if ln.in_table}
    assert "Senior Engineer, Acme GmbH" in flagged
    assert "Jane Doe" not in flagged


def test_scanned_page_has_no_text_and_a_note(scanned_pdf):
    doc = extract_pdf(scanned_pdf)
    assert doc.pages[0].char_count == 0
    assert doc.pages[0].image_area_ratio > 0.2
    if not doc.pages[0].ocr_used:
        assert any("OCR" in note for note in doc.notes)


def test_running_header_and_footer_are_flagged(multipage_pdf):
    doc = extract_pdf(multipage_pdf)
    flagged = [ln.text for ln in doc.lines if ln.in_header_footer]
    assert len(flagged) == 4  # contact header + page number, on both pages
    assert not any(ln.in_header_footer for ln in doc.lines if ln.text == "EXPERIENCE")


def test_icon_images_are_recorded(icon_pdf):
    doc = extract_pdf(icon_pdf)
    assert sum(img.is_icon_sized for img in doc.images) == 2


def test_accepts_bytes(single_column_pdf):
    doc = extract_pdf(single_column_pdf.read_bytes(), name="upload.pdf")
    assert doc.source == "upload.pdf"
    assert "Jane Doe" in doc.text


def test_garbage_input_raises():
    with pytest.raises(ValueError):
        extract_pdf(b"this is not a pdf")
