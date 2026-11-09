from app.ingestion.docx import extract_docx


def test_plain_docx_keeps_order_and_styles(plain_docx):
    doc = extract_docx(plain_docx)
    texts = [ln.text for ln in doc.lines]
    assert texts[0] == "Jane Doe"
    assert texts.index("Experience") < texts.index("Education") < texts.index("Skills")
    heading = next(ln for ln in doc.lines if ln.text == "Experience")
    assert heading.style == "Heading 1" and heading.bold
    bullet = next(ln for ln in doc.lines if ln.text.startswith("Led migration"))
    assert bullet.list_item
    assert not doc.tables and not doc.images and not doc.header_footer_text
    for ln in doc.lines:
        assert doc.text[ln.start:ln.end] == ln.text


def test_tricky_docx_flags_hidden_places(tricky_docx):
    doc = extract_docx(tricky_docx)
    by_text = {ln.text: ln for ln in doc.lines}

    # Contact info in the page header is extracted but flagged.
    assert by_text["Jane Doe"].in_header_footer
    assert any("jane.doe@example.com" in t for t in doc.header_footer_text)
    # Table cells are flagged and the table is recorded.
    assert by_text["Senior Software Engineer — Acme GmbH"].in_table
    assert doc.tables and doc.tables[0].rows == 2
    # Text box content is found (python-docx's Paragraph.text skips it).
    assert by_text["Open to relocation"].in_textbox
    # The photo is recorded, and it is not icon-sized.
    assert len(doc.images) == 1 and not doc.images[0].is_icon_sized


def test_symbol_font_glyphs_are_counted(symbol_docx):
    assert extract_docx(symbol_docx).symbol_glyphs == 1
